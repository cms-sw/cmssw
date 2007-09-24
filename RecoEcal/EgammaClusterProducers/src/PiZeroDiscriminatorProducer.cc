// system include files
#include <vector>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalPreshowerAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h" 
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include <fstream>
#include <sstream>

#include "RecoEcal/EgammaClusterProducers/interface/PiZeroDiscriminatorProducer.h"

// Class for Cluster Shape Algorithm
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"

#include "TFile.h"

using namespace std;
using namespace reco;
using namespace edm;
///----

PiZeroDiscriminatorProducer::PiZeroDiscriminatorProducer(const edm::ParameterSet& ps) {
  // use configuration file to setup input/output collection names
  // Parameters to identify the hit collections
  endcapHitProducer_   = ps.getParameter<std::string>("endcapHitProducer");
  endcapHitCollection_ = ps.getParameter<std::string>("endcapHitCollection");

  preshHitProducer_   = ps.getParameter<std::string>("preshRecHitProducer");
  preshHitCollection_ = ps.getParameter<std::string>("preshRecHitCollection");

  barrelHitProducer_   = ps.getParameter<std::string>("barrelHitProducer");  
  barrelHitCollection_ = ps.getParameter<std::string>("barrelHitCollection"); 

  photonCorrCollectionProducer_ = ps.getParameter<std::string>("corrPhoProducer");
  correctedPhotonCollection_ = ps.getParameter<std::string>("correctedPhotonCollection");

  float preshStripECut = ps.getParameter<double>("preshStripEnergyCut");
  int preshNst = ps.getParameter<int>("preshPi0Nstrip");

  std::map<std::string,double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted",ps.getParameter<bool>("posCalc_logweight")));
  providedParameters.insert(std::make_pair("T0_barl",ps.getParameter<double>("posCalc_t0_barl")));
  providedParameters.insert(std::make_pair("T0_endc",ps.getParameter<double>("posCalc_t0_endc")));
  providedParameters.insert(std::make_pair("T0_endcPresh",ps.getParameter<double>("posCalc_t0_endcPresh")));
  providedParameters.insert(std::make_pair("W0",ps.getParameter<double>("posCalc_w0")));
  providedParameters.insert(std::make_pair("X0",ps.getParameter<double>("posCalc_x0")));

  posCalculator_ = PositionCalc(providedParameters);
  shapeAlgo_ = ClusterShapeAlgo(providedParameters);

  PhotonPi0DiscriminatorAssociationMap_ = ps.getParameter<std::string>("Pi0Association");

  std::string debugString = ps.getParameter<std::string>("debugLevel");

  if      (debugString == "DEBUG")   debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pINFO;
  else                               debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pERROR;

  std::string tmpPath = ps.getUntrackedParameter<std::string>("pathToWeightFiles","RecoEcal/EgammaClusterProducers/data/");
  
  presh_pi0_algo = new EndcapPiZeroDiscriminatorAlgo(preshStripECut, preshNst, tmpPath.c_str(), debugL_pi0); 

  produces< reco::PhotonPi0DiscriminatorAssociationMap >(PhotonPi0DiscriminatorAssociationMap_);

  nEvt_ = 0;

}


PiZeroDiscriminatorProducer::~PiZeroDiscriminatorProducer() {
   delete presh_pi0_algo;
}


void PiZeroDiscriminatorProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  std::ostringstream ostr; // use this stream for all messages in produce

  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG )
       cout << "\n .......  Event " << evt.id() << " with Number = " <<  nEvt_+1
            << " is analyzing ....... " << endl << endl;

  edm::Handle< EcalRecHitCollection >   pRecHits;
  edm::Handle< reco::SuperClusterCollection > pSuperClusters;

  // get the ECAL -> Preshower geometry and topology:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;

   EcalPreshowerTopology topology(geoHandle);
   CaloSubdetectorTopology * topology_p = &topology;

  // fetch the Preshower product (RecHits)
  evt.getByLabel( preshHitProducer_, preshHitCollection_, pRecHits);
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product(); // EcalRecHitCollection hit_collection = *rhcHandle;
  if ( debugL_pi0 == EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout << "PiZeroDiscriminatorProducer: ### Total # of preshower RecHits: "
                                                          << rechits->size() << endl;
  // make the map of Preshower rechits:
  std::map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
     rechits_map.insert(std::make_pair(it->id(), *it));
  }
  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout
                                << "PiZeroDiscriminatorProducer: ### Preshower RecHits_map of size "
                                << rechits_map.size() <<" was created!" << endl;

  const CaloSubdetectorGeometry *geometry_pee;
  geometry_pee = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  EcalEndcapTopology topology_EE(geoHandle);
  //CaloSubdetectorTopology * topology_p = &topology;

  // fetch the ECAL Endcap product (RecHits)
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByLabel(endcapHitProducer_, endcapHitCollection_, rhcHandle);
  if (!(rhcHandle.isValid()))
    {
      cout << "Pi0rejection : could not get a handle on the EcalRecHitCollection!" << endl;
    }
  const EcalRecHitCollection *hit_collection = rhcHandle.product();


  const CaloSubdetectorGeometry *geometry_eb;
  geometry_eb = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  EcalBarrelTopology topology_EB(geoHandle);

  // fetch the ECAL Barrel product (RecHits)

  edm::Handle<EcalRecHitCollection> brhcHandle;
  evt.getByLabel(barrelHitProducer_, barrelHitCollection_, brhcHandle);
  if (!(brhcHandle.isValid()))
    {
      cout << "Pi0rejection : could not get a handle on the EcalRecHitCollection!" << endl;
    }
  const EcalRecHitCollection *bhit_collection = brhcHandle.product();

  std::auto_ptr<reco::PhotonPi0DiscriminatorAssociationMap> Pi0Assocs_p(new reco::PhotonPi0DiscriminatorAssociationMap);

  //make cycle over Photon Collection
  int Photon_index  = 0;
  Handle<reco::PhotonCollection> correctedPhotonHandle; 
  evt.getByLabel(photonCorrCollectionProducer_, correctedPhotonCollection_ , correctedPhotonHandle);
  const reco::PhotonCollection corrPhoCollection = *(correctedPhotonHandle.product());
  cout << " Photon Collection size : " << corrPhoCollection.size() << endl;
  for( reco::PhotonCollection::const_iterator  iPho = corrPhoCollection.begin(); iPho != corrPhoCollection.end(); iPho++) {
       if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
         cout << " Photon index : " << Photon_index 
                           << " with Energy = " <<  iPho->energy()
			   << " Et = " << iPho->energy()*sin(2*atan(exp(-iPho->eta())))
                           << " ETA = " << iPho->eta()
       		           << " PHI = " << iPho->phi() << endl;
       }
       reco::SuperClusterRef it_super = iPho->superCluster();

      float SC_Et   = it_super->energy()*sin(2*atan(exp(-it_super->eta())));
      float SC_eta  = it_super->eta();
      float SC_phi  = it_super->phi();

      if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
        cout << " superE = " << it_super->energy()
	          << "  superEt = " << it_super->energy()*sin(2*atan(exp(-it_super->eta()))) 
                  << " superETA = " << it_super->eta()
       		  << " superPHI = " << it_super->phi() << endl;
      }			   

      float nnoutput = -1.;
      if(fabs(SC_eta) >= 1.65 && fabs(SC_eta) <= 2.5) {  //  Use Preshower region only
          const GlobalPoint pointSC(it_super->x(),it_super->y(),it_super->z()); // get the centroid of the SC
          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout << "SC centroind = " << pointSC << endl;
          reco::BasicClusterRef BS_clus_Id = it_super->seed();
          double SC_seed_energy = it_super->seed()->energy();

          //ClusterShapeAlgo::Initialize(hit_collection, &geoHandle);
          reco::ClusterShape TestShape = shapeAlgo_.Calculate(*BS_clus_Id,hit_collection,geometry_pee,&topology_EE);
          double SC_seed_Shape_E1 = TestShape.eMax();
          double SC_seed_Shape_E3x3 = TestShape.e3x3();
          double SC_seed_Shape_E5x5 = TestShape.e5x5();
          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
            cout << "BC energy_max = " <<  SC_seed_energy << endl;
            cout << "ClusterShape  E1_max = " <<   SC_seed_Shape_E1 << endl;
            cout << "ClusterShape  E3x3_max = " <<   SC_seed_Shape_E3x3 <<  endl;
            cout << "ClusterShape  E5x5_max = " <<   SC_seed_Shape_E5x5 << endl;
          }
          DetId tmp_stripX = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 1);
          DetId tmp_stripY = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 2);
          ESDetId stripX = (tmp_stripX == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripX);
          ESDetId stripY = (tmp_stripY == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripY);

          std::vector<float> vout_stripE1 = presh_pi0_algo->findPreshVector(stripX, &rechits_map, topology_p);
          std::vector<float> vout_stripE2 = presh_pi0_algo->findPreshVector(stripY, &rechits_map, topology_p);

          bool valid_NNinput = presh_pi0_algo->calculateNNInputVariables(vout_stripE1, vout_stripE2,
                                                 SC_seed_Shape_E1, SC_seed_Shape_E3x3, SC_seed_Shape_E5x5);

          if(!valid_NNinput) {
            cout  << " PiZeroDiscProducer: Attention!!!!!  Not Valid NN input Variables Return " << endl;
	    continue;
	  }

          float* nn_input_var = presh_pi0_algo->get_input_vector();

          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
            cout  << " PiZeroDiscProducer: NN_input_vector = " ;
            for(int k1=0;k1<25;k1++) {
              cout  << nn_input_var[k1] << " " ;
            }
            cout  << endl;
	  }  

          nnoutput = presh_pi0_algo->GetNNOutput(SC_Et);

          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
               cout << "PreshowerPi0NNProducer:Event : " <<  evt.id()
	            << " SC id = " << Photon_index
		    << " with Pt = " << SC_Et
		    << " eta = " << SC_eta
		    << " phi = " << SC_phi
		    << " contains: " << it_super->clustersSize() << " BCs "
		    << " has NNout = " <<  nnoutput << endl;
         }
 	 Pi0Assocs_p->insert(edm::Ref<reco::PhotonCollection>(correctedPhotonHandle,Photon_index), nnoutput); 
	 
      } else if((fabs(SC_eta) <= 1.4442) || (fabs(SC_eta) < 1.65 && fabs(SC_eta) >= 1.566) || fabs(SC_eta) >= 2.5) {
 	 reco::BasicClusterRef BS_clus_Id = it_super->seed();
	 reco::ClusterShape TestShape;
         if(fabs(SC_eta) <= 1.4442) {
           TestShape = shapeAlgo_.Calculate(*BS_clus_Id,bhit_collection,geometry_eb,&topology_EB);
	 }  else { 
           TestShape = shapeAlgo_.Calculate(*BS_clus_Id,hit_collection,geometry_pee,&topology_EE);
         }
         double SC_seed_Shape_E1 = TestShape.eMax();
         double SC_seed_Shape_E3x3 = TestShape.e3x3();
         double SC_seed_Shape_E5x5 = TestShape.e5x5();
         double SC_seed_Shape_E2 = TestShape.e2nd();
         double SC_seed_Shape_cEE = TestShape.covEtaEta();
         double SC_seed_Shape_cEP = TestShape.covEtaPhi();
         double SC_seed_Shape_cPP = TestShape.covPhiPhi();
         double SC_seed_Shape_E2x2 = TestShape.e2x2();
         double SC_seed_Shape_E3x2 = TestShape.e3x2();
         double SC_seed_Shape_E3x2r = TestShape.e3x2Ratio();

         double SC_seed_Shape_xcog = TestShape.e2x5Right() - TestShape.e2x5Left();
         double SC_seed_Shape_ycog = TestShape.e2x5Bottom() - TestShape.e2x5Top();

         float SC_et = it_super->energy()*sin(2*atan(exp(-it_super->eta())));

         presh_pi0_algo->calculateBarrelNNInputVariables(SC_et, SC_seed_Shape_E1, SC_seed_Shape_E3x3,
					      SC_seed_Shape_E5x5, SC_seed_Shape_E2,
					      SC_seed_Shape_cEE, SC_seed_Shape_cEP,
					      SC_seed_Shape_cPP, SC_seed_Shape_E2x2,
					      SC_seed_Shape_E3x2, SC_seed_Shape_E3x2r,
					      SC_seed_Shape_xcog, SC_seed_Shape_ycog);

         float* nn_input_var = presh_pi0_algo->get_input_vector();

         nnoutput = presh_pi0_algo->GetBarrelNNOutput(SC_et);
         
	 if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
           cout  << " PiZeroDiscProducer: NN_barrel_Endcap_variables = " ;
           for(int k3=0;k3<12;k3++) {
             cout  << nn_input_var[k3] << " " ;
           }
           cout  << endl;

           cout << "EndcapPi0NNProducer:Event : " <<  evt.id()
	            << " SC id = " << Photon_index
		    << " with Pt = " << SC_Et
		    << " eta = " << SC_eta
		    << " phi = " << SC_phi
		    << " contains: " << it_super->clustersSize() << " BCs "
		    << " has NNout = " <<  nnoutput
	            << endl;
         }
 	 Pi0Assocs_p->insert(edm::Ref<reco::PhotonCollection>(correctedPhotonHandle,Photon_index), nnoutput);
      } else { Pi0Assocs_p->insert(edm::Ref<reco::PhotonCollection>(correctedPhotonHandle,Photon_index), -1.);}
      Photon_index++;
  } // end of cycle over Photons
  
  evt.put(Pi0Assocs_p,PhotonPi0DiscriminatorAssociationMap_);
  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout << "PhotonPi0DiscriminatorAssociationMap added to the event" << endl;

  nEvt_++;

  LogDebug("PiZeroDiscriminatorDebug") << ostr.str();


}

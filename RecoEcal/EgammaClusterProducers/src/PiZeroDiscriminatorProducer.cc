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
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalPreshowerAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include <fstream>
#include <sstream>

#include "RecoEcal/EgammaClusterProducers/interface/PiZeroDiscriminatorProducer.h"

// Class for Cluster Shape Algorithm
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
// ArisB 26/9/2007
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
// ArisE 26/9/2007
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"

using namespace std;
using namespace reco;
using namespace edm;
///----

PiZeroDiscriminatorProducer::PiZeroDiscriminatorProducer(const ParameterSet& ps) {
  // use configuration file to setup input/output collection names
<<<<<<< PiZeroDiscriminatorProducer.cc
=======
  // Parameters to identify the hit collections
  preshHitProducer_   = ps.getParameter<string>("preshRecHitProducer");
  preshHitCollection_ = ps.getParameter<string>("preshRecHitCollection");
>>>>>>> 1.8

<<<<<<< PiZeroDiscriminatorProducer.cc
  preshClusterShapeCollectionX_ = ps.getParameter<std::string>("preshClusterShapeCollectionX");
  preshClusterShapeCollectionY_ = ps.getParameter<std::string>("preshClusterShapeCollectionY");
  preshClusterShapeProducer_   = ps.getParameter<std::string>("preshClusterShapeProducer");

  photonCorrCollectionProducer_ = ps.getParameter<string>("corrPhoProducer");
  correctedPhotonCollection_ = ps.getParameter<string>("correctedPhotonCollection");
=======
  photonCorrCollectionProducer_ = ps.getParameter<string>("corrPhoProducer");
  correctedPhotonCollection_ = ps.getParameter<string>("correctedPhotonCollection");
>>>>>>> 1.8

  barrelClusterShapeMapProducer_   = ps.getParameter<string>("barrelClusterShapeMapProducer");
  barrelClusterShapeMapCollection_ = ps.getParameter<string>("barrelClusterShapeMapCollection");

  endcapClusterShapeMapProducer_   = ps.getParameter<string>("endcapClusterShapeMapProducer");
  endcapClusterShapeMapCollection_ = ps.getParameter<string>("endcapClusterShapeMapCollection");

  float preshStripECut = ps.getParameter<double>("preshStripEnergyCut");
  int preshNst = ps.getParameter<int>("preshPi0Nstrip");

  PhotonPi0DiscriminatorAssociationMap_ = ps.getParameter<string>("Pi0Association");

  string debugString = ps.getParameter<string>("debugLevel");

  if      (debugString == "DEBUG")   debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pINFO;
  else                               debugL_pi0 = EndcapPiZeroDiscriminatorAlgo::pERROR;

  string tmpPath = ps.getUntrackedParameter<string>("pathToWeightFiles","RecoEcal/EgammaClusterProducers/data/");
  
  presh_pi0_algo = new EndcapPiZeroDiscriminatorAlgo(preshStripECut, preshNst, tmpPath.c_str(), debugL_pi0); 

  produces< PhotonPi0DiscriminatorAssociationMap >(PhotonPi0DiscriminatorAssociationMap_);

  nEvt_ = 0;

}


PiZeroDiscriminatorProducer::~PiZeroDiscriminatorProducer() {
   delete presh_pi0_algo;
}


void PiZeroDiscriminatorProducer::produce(Event& evt, const EventSetup& es) {

  ostringstream ostr; // use this stream for all messages in produce

  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG )
       cout << "\n .......  Event " << evt.id() << " with Number = " <<  nEvt_+1
            << " is analyzing ....... " << endl << endl;

<<<<<<< PiZeroDiscriminatorProducer.cc
  // Get ES clusters in X plane
  Handle<reco::PreshowerClusterShapeCollection> pPreshowerShapeClustersX;
  evt.getByLabel(preshClusterShapeProducer_, preshClusterShapeCollectionX_, pPreshowerShapeClustersX);
  const reco::PreshowerClusterShapeCollection *clustersX = pPreshowerShapeClustersX.product();
  cout << "\n pPreshowerShapeClustersX->size() = " << clustersX->size() << endl;

  // Get ES clusters in Y plane
  Handle<reco::PreshowerClusterShapeCollection> pPreshowerShapeClustersY;
  evt.getByLabel(preshClusterShapeProducer_, preshClusterShapeCollectionY_, pPreshowerShapeClustersY);
  const reco::PreshowerClusterShapeCollection *clustersY = pPreshowerShapeClustersY.product();
  cout << "\n pPreshowerShapeClustersY->size() = " << clustersY->size() << endl;

// Get association maps linking BasicClusters to ClusterShape
  Handle<BasicClusterShapeAssociationCollection> barrelClShpHandle;
  evt.getByLabel(barrelClusterShapeMapProducer_, barrelClusterShapeMapCollection_,barrelClShpHandle);
  const BasicClusterShapeAssociationCollection& barrelClShpMap = *barrelClShpHandle;
  
  Handle<BasicClusterShapeAssociationCollection> endcapClShpHandle;
  evt.getByLabel(endcapClusterShapeMapProducer_, endcapClusterShapeMapCollection_, endcapClShpHandle);
  const BasicClusterShapeAssociationCollection& endcapClShpMap = *endcapClShpHandle;
  
  auto_ptr<PhotonPi0DiscriminatorAssociationMap> Pi0Assocs_p(new PhotonPi0DiscriminatorAssociationMap);
=======
  Handle< EcalRecHitCollection >   pRecHits;
  Handle< SuperClusterCollection > pSuperClusters;

  // get the ECAL -> Preshower geometry and topology:
  ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;

   EcalPreshowerTopology topology(geoHandle);
   CaloSubdetectorTopology * topology_p = &topology;

  // fetch the Preshower product (RecHits)
  evt.getByLabel( preshHitProducer_, preshHitCollection_, pRecHits);
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product(); 
  if ( debugL_pi0 == EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout << "PiZeroDiscriminatorProducer: ### Total # of preshower RecHits: "
                                                          << rechits->size() << endl;
  // make the map of Preshower rechits:
  map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
     rechits_map.insert(make_pair(it->id(), *it));
  }
  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout
                                << "PiZeroDiscriminatorProducer: ### Preshower RecHits_map of size "
                                << rechits_map.size() <<" was created!" << endl;

  auto_ptr<PhotonPi0DiscriminatorAssociationMap> Pi0Assocs_p(new PhotonPi0DiscriminatorAssociationMap);

// Get association maps linking BasicClusters to ClusterShape
  Handle<BasicClusterShapeAssociationCollection> barrelClShpHandle;
  evt.getByLabel(barrelClusterShapeMapProducer_, barrelClusterShapeMapCollection_,barrelClShpHandle);
  const BasicClusterShapeAssociationCollection& barrelClShpMap = *barrelClShpHandle;
  
  Handle<BasicClusterShapeAssociationCollection> endcapClShpHandle;
  evt.getByLabel(endcapClusterShapeMapProducer_, endcapClusterShapeMapCollection_, endcapClShpHandle);
  const BasicClusterShapeAssociationCollection& endcapClShpMap = *endcapClShpHandle;
>>>>>>> 1.8

  //make cycle over Photon Collection
  int Photon_index  = 0;
  Handle<PhotonCollection> correctedPhotonHandle; 
  evt.getByLabel(photonCorrCollectionProducer_, correctedPhotonCollection_ , correctedPhotonHandle);
  const PhotonCollection corrPhoCollection = *(correctedPhotonHandle.product());
  cout << " Photon Collection size : " << corrPhoCollection.size() << endl;
<<<<<<< PiZeroDiscriminatorProducer.cc
  for( PhotonCollection::const_iterator  iPho = corrPhoCollection.begin(); iPho != corrPhoCollection.end(); iPho++) {
       float Phot_R9 = iPho->r9();
=======
  for( PhotonCollection::const_iterator  iPho = corrPhoCollection.begin(); iPho != corrPhoCollection.end(); iPho++) {
>>>>>>> 1.8
       if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
         cout << " Photon index : " << Photon_index 
                           << " with Energy = " <<  iPho->energy()
			   << " Et = " << iPho->energy()*sin(2*atan(exp(-iPho->eta())))
                           << " ETA = " << iPho->eta()
       		           << " PHI = " << iPho->phi() << endl;
       }
       SuperClusterRef it_super = iPho->superCluster();

      float SC_Et   = it_super->energy()*sin(2*atan(exp(-it_super->eta())));
      float SC_eta  = it_super->eta();
      float SC_phi  = it_super->phi();

      if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
        cout << " superE = " << it_super->energy()
	          << "  superEt = " << it_super->energy()*sin(2*atan(exp(-it_super->eta()))) 
                  << " superETA = " << it_super->eta()
       		  << " superPHI = " << it_super->phi() << endl;
      }			   

      //  New way to get ClusterShape info
      BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
      // Find the entry in the map corresponding to the seed BasicCluster of the SuperCluster
      DetId id = it_super->seed()->getHitsByDetId()[0];

      float nnoutput = -1.;
      if(fabs(SC_eta) >= 1.65 && fabs(SC_eta) <= 2.5) {  //  Use Preshower region only
          const GlobalPoint pointSC(it_super->x(),it_super->y(),it_super->z()); // get the centroid of the SC
          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout << "SC centroind = " << pointSC << endl;
          double SC_seed_energy = it_super->seed()->energy();

          seedShpItr = endcapClShpMap.find(it_super->seed());         
          const ClusterShapeRef& seedShapeRef = seedShpItr->val; // Get the ClusterShapeRef corresponding to the BasicCluster
          double SC_seed_Shape_E1 = seedShapeRef->eMax();
          double SC_seed_Shape_E3x3 = seedShapeRef->e3x3();
          double SC_seed_Shape_E5x5 = seedShapeRef->e5x5();
          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
            cout << "BC energy_max = " <<  SC_seed_energy << endl;
<<<<<<< PiZeroDiscriminatorProducer.cc
            cout << "ClusterShape  E1_max_New = " <<   SC_seed_Shape_E1 << endl;
            cout << "ClusterShape  E3x3_max_New = " <<   SC_seed_Shape_E3x3 <<  endl;
            cout << "ClusterShape  E5x5_max_New = " <<   SC_seed_Shape_E5x5 << endl;
          }           
// Get the Preshower 2-planes energy vectors associated with the given SC
          vector<float> vout_stripE1;
	  vector<float> vout_stripE2;
          for(reco::PreshowerClusterShapeCollection::const_iterator esClus = clustersX->begin();
                                                       esClus !=clustersX->end(); esClus++) {
             if( it_super == esClus->superCluster()) {
	        
               vout_stripE1 = esClus->getStripEnergies();
	       
             }
          }
	  for(reco::PreshowerClusterShapeCollection::const_iterator esClus = clustersY->begin();
                                                       esClus !=clustersY->end(); esClus++) {
            if( it_super == esClus->superCluster()) {				       
	    
               vout_stripE2 = esClus->getStripEnergies();
	    }  
          }
          if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
            cout  << "PiZeroDiscriminatorProducer : ES_input_vector = " ;
            for(int k1=0;k1<11;k1++) {
              cout  << vout_stripE1[k1] << " " ;
            }
            for(int k1=0;k1<11;k1++) {
              cout  << vout_stripE2[k1] << " " ;
            }
            cout  << endl;
          }
	  
=======
            cout << "ClusterShape  E1_max_New = " <<   SC_seed_Shape_E1 << endl;
            cout << "ClusterShape  E3x3_max_New = " <<   SC_seed_Shape_E3x3 <<  endl;
            cout << "ClusterShape  E5x5_max_New = " <<   SC_seed_Shape_E5x5 << endl;
          }           
// Get the Preshower 2-planes RecHit vectors associated with the given SC
          DetId tmp_stripX = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 1);
          DetId tmp_stripY = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(pointSC, 2);
          ESDetId stripX = (tmp_stripX == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripX);
          ESDetId stripY = (tmp_stripY == DetId(0)) ? ESDetId(0) : ESDetId(tmp_stripY);

          vector<float> vout_stripE1 = presh_pi0_algo->findPreshVector(stripX, &rechits_map, topology_p);
          vector<float> vout_stripE2 = presh_pi0_algo->findPreshVector(stripY, &rechits_map, topology_p);

>>>>>>> 1.8
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
                    << " R9 = " << Phot_R9
		    << " has NNout = " <<  nnoutput << endl;
         }
 	 Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,Photon_index), nnoutput); 
	 
      } else if((fabs(SC_eta) <= 1.4442) || (fabs(SC_eta) < 1.65 && fabs(SC_eta) >= 1.566) || fabs(SC_eta) >= 2.5) {

         if (id.subdetId() == EcalBarrel) {
           seedShpItr = barrelClShpMap.find(it_super->seed());
         } else {
           seedShpItr = endcapClShpMap.find(it_super->seed());
         }
	  
         const ClusterShapeRef& seedShapeRef = seedShpItr->val; // Get the ClusterShapeRef corresponding to the BasicCluster
	  
         double SC_seed_Shape_E1 = seedShapeRef->eMax();
         double SC_seed_Shape_E3x3 = seedShapeRef->e3x3();
         double SC_seed_Shape_E5x5 = seedShapeRef->e5x5();
         double SC_seed_Shape_E2 = seedShapeRef->e2nd();
         double SC_seed_Shape_cEE = seedShapeRef->covEtaEta();
         double SC_seed_Shape_cEP = seedShapeRef->covEtaPhi();
         double SC_seed_Shape_cPP = seedShapeRef->covPhiPhi();
         double SC_seed_Shape_E2x2 = seedShapeRef->e2x2();
         double SC_seed_Shape_E3x2 = seedShapeRef->e3x2();
         double SC_seed_Shape_E3x2r = seedShapeRef->e3x2Ratio();

         double SC_seed_Shape_xcog = seedShapeRef->e2x5Right() - seedShapeRef->e2x5Left();
         double SC_seed_Shape_ycog = seedShapeRef->e2x5Bottom() - seedShapeRef->e2x5Top();
         if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) {
            cout << "ClusterShape  E1_max_New = " <<   SC_seed_Shape_E1 << endl;
            cout << "ClusterShape  E3x3_max_New = " <<   SC_seed_Shape_E3x3 <<  endl;
            cout << "ClusterShape  E5x5_max_New = " <<   SC_seed_Shape_E5x5 << endl;
            cout << "ClusterShape  E2_max_New = " <<   SC_seed_Shape_E2 << endl;
            cout << "ClusterShape  EE_max_New = " <<   SC_seed_Shape_cEE <<  endl;
            cout << "ClusterShape  EP_max_New = " <<   SC_seed_Shape_cEP << endl;	    
            cout << "ClusterShape  PP_max_New = " <<   SC_seed_Shape_cPP << endl;
            cout << "ClusterShape  E2x2_max_New = " <<   SC_seed_Shape_E2x2 <<  endl;
            cout << "ClusterShape  E3x2_max_New = " <<   SC_seed_Shape_E3x2 << endl;
            cout << "ClusterShape  E3x2r_max_New = " <<   SC_seed_Shape_E3x2r << endl;
            cout << "ClusterShape  xcog_max_New = " <<   SC_seed_Shape_xcog << endl;
            cout << "ClusterShape  ycog_max_New = " <<   SC_seed_Shape_ycog << endl;	    	    
         }    

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
		    << " R9 = " << Phot_R9
		    << " has NNout = " <<  nnoutput
	            << endl;
         }
 	 Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,Photon_index), nnoutput);
      } else { Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,Photon_index), -1.);}
      Photon_index++;
  } // end of cycle over Photons
  
  evt.put(Pi0Assocs_p,PhotonPi0DiscriminatorAssociationMap_);
  if ( debugL_pi0 <= EndcapPiZeroDiscriminatorAlgo::pDEBUG ) cout << "PhotonPi0DiscriminatorAssociationMap added to the event" << endl;

  nEvt_++;

  LogDebug("PiZeroDiscriminatorDebug") << ostr.str();


}

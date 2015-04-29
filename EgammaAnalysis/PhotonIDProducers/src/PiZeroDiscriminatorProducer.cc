// system include files
#include <vector>

// user include files

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <fstream>
#include <sstream>

#include "EgammaAnalysis/PhotonIDProducers/interface/PiZeroDiscriminatorProducer.h"

// Class for Cluster Shape Algorithm
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"

// to compute on-the-fly cluster shapes
//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


using namespace std;
using namespace reco;
using namespace edm;
///----

PiZeroDiscriminatorProducer::PiZeroDiscriminatorProducer(const ParameterSet& ps) {
  // use configuration file to setup input/output collection names

  pPreshowerShapeClustersXToken_ = consumes<PreshowerClusterShapeCollection>(edm::InputTag(ps.getParameter<std::string>("preshClusterShapeProducer"), ps.getParameter<std::string>("preshClusterShapeCollectionX")));
  pPreshowerShapeClustersYToken_ = consumes<PreshowerClusterShapeCollection>(edm::InputTag(ps.getParameter<std::string>("preshClusterShapeProducer"), ps.getParameter<std::string>("preshClusterShapeCollectionY")));

  correctedPhotonToken_ = consumes<PhotonCollection>(edm::InputTag(ps.getParameter<string>("corrPhoProducer"), ps.getParameter<string>("correctedPhotonCollection")));

  barrelRecHitCollection_ = ps.getParameter<edm::InputTag>("barrelRecHitCollection");
  barrelRecHitCollectionToken_ = consumes<EcalRecHitCollection>(barrelRecHitCollection_);
  endcapRecHitCollection_ = ps.getParameter<edm::InputTag>("endcapRecHitCollection");
  endcapRecHitCollectionToken_ = consumes<EcalRecHitCollection>(endcapRecHitCollection_);

  EScorr_ = ps.getParameter<int>("EScorr");

  preshNst_ = ps.getParameter<int>("preshPi0Nstrip");

  preshStripECut_ = ps.getParameter<double>("preshStripEnergyCut");

  w0_ = ps.getParameter<double>("w0");

  PhotonPi0DiscriminatorAssociationMap_ = ps.getParameter<string>("Pi0Association");

  string debugString = ps.getParameter<string>("debugLevel");

  if      (debugString == "DEBUG")   debugL_pi0 = pDEBUG;
  else if (debugString == "INFO")    debugL_pi0 = pINFO;
  else                               debugL_pi0 = pERROR;

  string tmpPath = ps.getUntrackedParameter<string>("pathToWeightFiles","RecoEcal/EgammaClusterProducers/data/");

  presh_pi0_algo = new EndcapPiZeroDiscriminatorAlgo(preshStripECut_, preshNst_, tmpPath.c_str());

  produces< PhotonPi0DiscriminatorAssociationMap >(PhotonPi0DiscriminatorAssociationMap_);


  nEvt_ = 0;

}


PiZeroDiscriminatorProducer::~PiZeroDiscriminatorProducer() {
   delete presh_pi0_algo;
}


void PiZeroDiscriminatorProducer::produce(Event& evt, const EventSetup& es) {

  ostringstream ostr; // use this stream for all messages in produce

  if ( debugL_pi0 <= pDEBUG )
       cout << "\n PiZeroDiscriminatorProducer: .......  Event " << evt.id() << " with Number = " <<  nEvt_+1
            << " is analyzing ....... " << endl << endl;

  // Get ES clusters in X plane
  Handle<reco::PreshowerClusterShapeCollection> pPreshowerShapeClustersX;
  evt.getByToken(pPreshowerShapeClustersXToken_, pPreshowerShapeClustersX);
  const reco::PreshowerClusterShapeCollection *clustersX = pPreshowerShapeClustersX.product();
  if ( debugL_pi0 <= pDEBUG ) {
    cout << "\n PiZeroDiscriminatorProducer: pPreshowerShapeClustersX->size() = " << clustersX->size() << endl;
  }
  // Get ES clusters in Y plane
  Handle<reco::PreshowerClusterShapeCollection> pPreshowerShapeClustersY;
  evt.getByToken(pPreshowerShapeClustersYToken_, pPreshowerShapeClustersY);
  const reco::PreshowerClusterShapeCollection *clustersY = pPreshowerShapeClustersY.product();
  if ( debugL_pi0 <= pDEBUG ) {
    cout << "\n PiZeroDiscriminatorProducer: pPreshowerShapeClustersY->size() = " << clustersY->size() << endl;
  }

  Handle< EcalRecHitCollection > pEBRecHits;
  evt.getByToken( barrelRecHitCollectionToken_, pEBRecHits );
  const EcalRecHitCollection *ebRecHits = pEBRecHits.product();

  Handle< EcalRecHitCollection > pEERecHits;
  evt.getByToken( endcapRecHitCollectionToken_, pEERecHits );
  const EcalRecHitCollection *eeRecHits = pEERecHits.product();

  ESHandle<CaloGeometry> pGeometry;
  es.get<CaloGeometryRecord>().get(pGeometry);
  const CaloGeometry *geometry = pGeometry.product();

  ESHandle<CaloTopology> pTopology;
  es.get<CaloTopologyRecord>().get(pTopology);
  const CaloTopology *topology = pTopology.product();

  //make cycle over Photon Collection
  Handle<PhotonCollection> correctedPhotonHandle;
  evt.getByToken(correctedPhotonToken_, correctedPhotonHandle);
  const PhotonCollection corrPhoCollection = *(correctedPhotonHandle.product());
  if ( debugL_pi0 <= pDEBUG ) {
    cout << " PiZeroDiscriminatorProducer: Photon Collection size : " << corrPhoCollection.size() << endl;
  }

  auto_ptr<PhotonPi0DiscriminatorAssociationMap> Pi0Assocs_p(new PhotonPi0DiscriminatorAssociationMap(correctedPhotonHandle));

  for( PhotonCollection::const_iterator  iPho = corrPhoCollection.begin(); iPho != corrPhoCollection.end(); iPho++) {

      float Phot_En   = iPho->energy();
      float Phot_Et   = Phot_En*sin(2*atan(exp(-iPho->eta())));
      float Phot_eta  = iPho->eta();
      float Phot_phi  = iPho->phi();
      float Phot_R9   = iPho->r9();

      if ( debugL_pi0 <= pDEBUG ) {
         cout << " PiZeroDiscriminatorProducer: Photon index : " << iPho - corrPhoCollection.begin()
                           << " with Energy = " <<  Phot_En
			   << " Et = " << Phot_Et
                           << " ETA = " << Phot_eta
       		           << " PHI = " << Phot_phi
			   << " R9 = " << Phot_R9 << endl;
      }
      SuperClusterRef it_super = iPho->superCluster();

      float SC_En   = it_super->energy();
      float SC_Et   = SC_En*sin(2*atan(exp(-it_super->eta())));
      float SC_eta  = it_super->eta();
      float SC_phi  = it_super->phi();

      if ( debugL_pi0 <= pDEBUG ) {
        cout << " PiZeroDiscriminatorProducer: superE = " << SC_En
	          << " superEt = " << SC_Et
                  << " superETA = " << SC_eta
       		  << " superPHI = " << SC_phi << endl;
      }

      //  New way to get ClusterShape info
      // Find the entry in the map corresponding to the seed BasicCluster of the SuperCluster
      // DetId id = it_super->seed()->hitsAndFractions()[0].first;

      // get on-the-fly the cluster shapes
//      EcalClusterLazyTools lazyTool( evt, es, barrelRecHitCollection_, endcapRecHitCollection_ );

      float nnoutput = -1.;
      if(fabs(SC_eta) >= 1.65 && fabs(SC_eta) <= 2.5) {  //  Use Preshower region only
          const GlobalPoint pointSC(it_super->x(),it_super->y(),it_super->z()); // get the centroid of the SC
          if ( debugL_pi0 <= pDEBUG ) cout << "SC centroind = " << pointSC << endl;
          double SC_seed_energy = it_super->seed()->energy();

          const CaloClusterPtr  seed = it_super->seed();

	  EcalClusterTools::eMax( *seed, ebRecHits );

          double SC_seed_Shape_E1 = EcalClusterTools::eMax( *seed, eeRecHits );
          double SC_seed_Shape_E3x3 = EcalClusterTools::e3x3( *seed, eeRecHits, topology );
          double SC_seed_Shape_E5x5 = EcalClusterTools::e5x5( *seed, eeRecHits, topology );

          if ( debugL_pi0 <= pDEBUG ) {
            cout << "PiZeroDiscriminatorProducer: ( SeedBC_energy, E1, E3x3, E5x5) = "
	         <<  SC_seed_energy << " "
                 <<  SC_seed_Shape_E1 <<  " "
                 <<  SC_seed_Shape_E3x3 <<  " "
                 <<  SC_seed_Shape_E5x5 << endl;
          }

// Get the Preshower 2-planes energy vectors associated with the given SC
          vector<float> vout_stripE1;
	  vector<float> vout_stripE2;
          for(reco::PreshowerClusterShapeCollection::const_iterator esClus = clustersX->begin();
                                                       esClus !=clustersX->end(); esClus++) {
             SuperClusterRef sc_ref = esClus->superCluster();
             float dR = sqrt((SC_eta-sc_ref->eta())*(SC_eta-sc_ref->eta()) +
	                     (SC_phi-sc_ref->phi())*(SC_phi-sc_ref->phi()));
             if(dR < 0.01 ) {

	       vout_stripE1 = esClus->getStripEnergies();

             }
          }
	  for(reco::PreshowerClusterShapeCollection::const_iterator esClus = clustersY->begin();
                                                       esClus !=clustersY->end(); esClus++) {
            SuperClusterRef sc_ref = esClus->superCluster();
	    float dR = sqrt((SC_eta-sc_ref->eta())*(SC_eta-sc_ref->eta()) +
	                     (SC_phi-sc_ref->phi())*(SC_phi-sc_ref->phi()));
             if(dR < 0.01 ) {

               vout_stripE2 = esClus->getStripEnergies();

	    }
          }

          if(vout_stripE1.size() == 0 || vout_stripE2.size() == 0 ) {
            if ( debugL_pi0 <= pDEBUG )
	            cout  << " PiZeroDiscriminatorProducer: Attention!!!!!  Not Valid ES NN input Variables Return NNout = -1" << endl;
	    Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,iPho - corrPhoCollection.begin()), nnoutput);
            continue;
	  }

          if ( debugL_pi0 <= pDEBUG ) {
            cout  << "PiZeroDiscriminatorProducer : vout_stripE1.size = " << vout_stripE1.size()
	          << " vout_stripE2.size = " << vout_stripE2.size() << endl;
            cout  << "PiZeroDiscriminatorProducer : ES_input_vector = " ;
            for(int k1=0;k1<11;k1++) {
              cout  << vout_stripE1[k1] << " " ;
            }
            for(int k1=0;k1<11;k1++) {
              cout  << vout_stripE2[k1] << " " ;
            }
            cout  << endl;
          }

          bool valid_NNinput = presh_pi0_algo->calculateNNInputVariables(vout_stripE1, vout_stripE2,
                                                 SC_seed_Shape_E1, SC_seed_Shape_E3x3, SC_seed_Shape_E5x5, EScorr_);

          if(!valid_NNinput) {
            if ( debugL_pi0 <= pDEBUG )
	           cout  << " PiZeroDiscriminatorProducer: Attention!!!!!  Not Valid ES NN input Variables Return NNout = -1" << endl;
	    Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,iPho - corrPhoCollection.begin()), nnoutput);
	    continue;
	  }

          float* nn_input_var = presh_pi0_algo->get_input_vector();

          if ( debugL_pi0 <= pDEBUG ) {
            cout  << " PiZeroDiscriminatorProducer: NN_ESEndcap_input_vector+Et+Eta+Phi+R9 = " ;
            for(int k1=0;k1<25;k1++) {
              cout  << nn_input_var[k1] << " " ;
            }
            cout  << SC_Et << " " << SC_eta << " " << SC_phi << " " << Phot_R9  << endl;
	  }

          nnoutput = presh_pi0_algo->GetNNOutput(SC_Et);

          if ( debugL_pi0 <= pDEBUG ) {
               cout << " PiZeroDiscriminatorProducer: Event : " <<  evt.id()
	            << " SC id = " << iPho - corrPhoCollection.begin()
		    << " with Pt = " << SC_Et
		    << " eta = " << SC_eta
		    << " phi = " << SC_phi
		    << " contains: " << it_super->clustersSize() << " BCs "
		    << " has NNout = " <<  nnoutput << endl;
         }

 	 Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,iPho - corrPhoCollection.begin()), nnoutput);

      } else if((fabs(SC_eta) <= 1.4442) || (fabs(SC_eta) < 1.65 && fabs(SC_eta) >= 1.566) || fabs(SC_eta) >= 2.5) {

         const CaloClusterPtr seed = it_super->seed();

         double SC_seed_Shape_E1 = EcalClusterTools::eMax( *seed, ebRecHits );
         double SC_seed_Shape_E3x3 = EcalClusterTools::e3x3( *seed, ebRecHits, topology );
         double SC_seed_Shape_E5x5 = EcalClusterTools::e5x5( *seed, ebRecHits, topology );
         double SC_seed_Shape_E2 = EcalClusterTools::e2nd( *seed, ebRecHits );

	 std::vector<float> vCov = EcalClusterTools::covariances( *seed, ebRecHits , topology, geometry, w0_ );

         double SC_seed_Shape_cEE = vCov[0];
         double SC_seed_Shape_cEP = vCov[1];
         double SC_seed_Shape_cPP = vCov[2];

         double SC_seed_Shape_E2x2 = EcalClusterTools::e2x2( *seed, ebRecHits, topology );
         double SC_seed_Shape_E3x2 = EcalClusterTools::e3x2( *seed, ebRecHits, topology );

	 double SC_seed_Shape_E3x2r = 0.0;
	 double SC_seed_Shape_ELeft = EcalClusterTools::eLeft( *seed, ebRecHits, topology );
	 double SC_seed_Shape_ERight = EcalClusterTools::eRight( *seed, ebRecHits, topology );
	 double SC_seed_Shape_ETop = EcalClusterTools::eTop( *seed, ebRecHits, topology );
	 double SC_seed_Shape_EBottom = EcalClusterTools::eBottom( *seed, ebRecHits, topology );

         double DA = SC_seed_Shape_E2x2 - SC_seed_Shape_E2 - SC_seed_Shape_E1;

         if(SC_seed_Shape_E2==SC_seed_Shape_ETop || SC_seed_Shape_E2==SC_seed_Shape_EBottom) {
	   if( SC_seed_Shape_ELeft > SC_seed_Shape_ERight ) {
	     SC_seed_Shape_E3x2r = (DA - SC_seed_Shape_ELeft)/(0.25+SC_seed_Shape_ELeft);
	   } else {
	     SC_seed_Shape_E3x2r = (DA - SC_seed_Shape_ERight)/(0.25+SC_seed_Shape_ERight);
	   }

	 } else if(SC_seed_Shape_E2==SC_seed_Shape_ELeft || SC_seed_Shape_E2==SC_seed_Shape_ERight) {

	   if( SC_seed_Shape_ETop > SC_seed_Shape_EBottom ) {
	     SC_seed_Shape_E3x2r = (DA - SC_seed_Shape_ETop)/(0.25+SC_seed_Shape_ETop);
	   } else {
	     SC_seed_Shape_E3x2r = (DA - SC_seed_Shape_EBottom)/(0.25+SC_seed_Shape_EBottom);
	   }

         }

         double SC_seed_Shape_xcog = EcalClusterTools::eRight( *seed, ebRecHits, topology ) - EcalClusterTools::e2x5Left( *seed, ebRecHits, topology );
         double SC_seed_Shape_ycog = EcalClusterTools::e2x5Top( *seed, ebRecHits, topology ) - EcalClusterTools::e2x5Bottom( *seed, ebRecHits, topology );


         if ( debugL_pi0 <= pDEBUG ) {
            cout << "PiZeroDiscriminatorProduce: lazyTool  (E1,E3x3,E5x5,E2,cEE,cEP,cPP,E2x2,E3x2_E3x2r,Xcog,Ycog,E2x5Bottom,E2x5Top,Et,Eta,PhiR9) = ( "
	         <<   SC_seed_Shape_E1 << " "
                 <<   SC_seed_Shape_E3x3 << " "
                 <<   SC_seed_Shape_E5x5 << " "
                 <<   SC_seed_Shape_E2 << " "
                 <<   SC_seed_Shape_cEE <<  " "
                 <<   SC_seed_Shape_cEP << " "
                 <<   SC_seed_Shape_cPP << " "
                 <<   SC_seed_Shape_E2x2 <<  " "
                 <<   SC_seed_Shape_E3x2 << " "
                 <<   SC_seed_Shape_E3x2r << " "
                 <<   SC_seed_Shape_xcog <<  " "
                 <<   SC_seed_Shape_ycog << " "
		 <<   EcalClusterTools::e2x5Bottom( *seed, ebRecHits, topology ) << " "
		 <<   EcalClusterTools::e2x5Top( *seed, ebRecHits, topology ) << " "
		 <<   SC_Et << " "
		 <<   SC_eta << " "
		 <<   SC_phi << " "
	         <<   Phot_R9 << " )" << endl;
         }

         float SC_et = it_super->energy()*sin(2*atan(exp(-it_super->eta())));

         presh_pi0_algo->calculateBarrelNNInputVariables(SC_et, SC_seed_Shape_E1, SC_seed_Shape_E3x3,
					      SC_seed_Shape_E5x5, SC_seed_Shape_E2,
					      SC_seed_Shape_cEE, SC_seed_Shape_cEP,
					      SC_seed_Shape_cPP, SC_seed_Shape_E2x2,
					      SC_seed_Shape_E3x2, SC_seed_Shape_E3x2r,
					      SC_seed_Shape_xcog, SC_seed_Shape_ycog);

         float* nn_input_var = presh_pi0_algo->get_input_vector();

  	 if ( debugL_pi0 <= pDEBUG ) {
           cout  << " PiZeroDiscriminatorProducer : NN_barrel_nonESEndcap_variables+Et+Eta+Phi+R9 = " ;
           for(int k3=0;k3<12;k3++) {
             cout  << nn_input_var[k3] << " " ;
           }
           cout  << SC_Et << " " << SC_eta << " " << SC_phi << " " << Phot_R9 << endl;

         }

         nnoutput = presh_pi0_algo->GetBarrelNNOutput(SC_et);


         if ( debugL_pi0 <= pDEBUG ) {
           cout << "PiZeroDiscriminatorProducer : Event : " <<  evt.id()
	            << " SC id = " << iPho - corrPhoCollection.begin()
		    << " with Pt = " << SC_Et
		    << " eta = " << SC_eta
		    << " phi = " << SC_phi
		    << " contains: " << it_super->clustersSize() << " BCs "
		    << " has NNout = " <<  nnoutput
	            << endl;
         }

 	 Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,iPho - corrPhoCollection.begin()), nnoutput);
      } else { Pi0Assocs_p->insert(Ref<PhotonCollection>(correctedPhotonHandle,iPho - corrPhoCollection.begin()), -1.);}
  } // end of cycle over Photons

  evt.put(Pi0Assocs_p,PhotonPi0DiscriminatorAssociationMap_);
  if ( debugL_pi0 <= pDEBUG ) cout << "PiZeroDiscriminatorProducer: PhotonPi0DiscriminatorAssociationMap added to the event" << endl;

  nEvt_++;

  LogDebug("PiZeroDiscriminatorDebug") << ostr.str();


}

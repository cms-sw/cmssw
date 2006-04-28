#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterProducer.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Geometry/Point3D.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "RecoCaloTools/Navigation/test/stubs/CaloNavigationAnalyzer.h"
#include <vector>

PreshowerClusterProducer::PreshowerClusterProducer(const edm::ParameterSet& ps) {

  // use configuration file to setup input/output collection names 
  hitProducer_   = ps.getParameter<std::string>("ESRecHitProducer");
  hitCollection_ = ps.getParameter<std::string>("ESRecHitCollection");

  // Name of a SuperClusterCollection to make associations:
  SClusterCollection_ = ps.getParameter<std::string>("SuperClusterCollection");

  // configure the algorithm via ParameterSet
  // algo parameters:
  double StripEnergyCut   = ps.getUntrackedParameter<double>("PreshStripEnergyCut",0.);
  double ClusterEnergyCut = ps.getUntrackedParameter<double>("PreshClusterEnergyCut",0.);
     int NStripCut        = ps.getUntrackedParameter<int>("PreshSeededNstrip",15);
  PreshNclust_            = ps.getUntrackedParameter<int>("PreshNclust",4);
  // calibration parameters:
  calib_plane1_ = ps.getUntrackedParameter<double>("PreshCalibPlane_X",1.0);
  calib_plane2_ = ps.getUntrackedParameter<double>("PreshCalibPlane_Y",0.70);
  miptogev_     = ps.getUntrackedParameter<double>("PreshCalibMIPtoGeV",0.024);

  // Output collections:
  clusterCollection1_ = ps.getParameter<std::string>("PreshClusterCollection_X");
  clusterCollection2_ = ps.getParameter<std::string>("PreshClusterCollection_Y");

  presh_algo_ = new PreshowerClusterAlgo(StripEnergyCut,ClusterEnergyCut,NStripCut);
  
}

PreshowerClusterProducer::~PreshowerClusterProducer() {
 delete presh_algo_;
}


void PreshowerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  using namespace edm;

  Handle< EcalRecHitCollection >   pRecHits;
  Handle< reco::SuperClusterCollection > pSuperClusters;

  edm::ESHandle<CaloTopology> theCaloTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopology);     

  // fetch the product (RecHits)
  try {
    evt.getByLabel( hitProducer_, hitCollection_, pRecHits);
  } catch ( cms::Exception& ex ) {
    edm::LogError("PreshowerClusterProducerError") << "Error! can't get the product " << hitCollection_.c_str();
  }
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product();
  edm::LogInfo("PreshowerClusterProducerInfo") << "total # of rechits: " << rechits->size();

 // fetch the product (pSuperClusters)
  try {
    evt.getByLabel( SClusterCollection_, pSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("PreshowerClusterProducerError") << "Error! can't get the product " << SClusterCollection_.c_str();
  }
  // pointer to the object in the product
  const reco::SuperClusterCollection* SClusts = pSuperClusters.product();
  edm::LogInfo("PreshowerClusterProducerInfo") << "total #  superclusters: " << SClusts->size();

  // Initialize preshower rechits:
  presh_algo_->PreshHitsInit( *rechits );

  reco::PreshowerClusterCollection clusters1, clusters2;   
  //make cycle over super clusters
  reco::SuperClusterCollection::iterator it_super;
  int isc = 0;
  for ( it_super=SClusts->begin();  it_super!=SClusts->end(); it_super++ ) 
     {     
       float e1=0;
       float e2=0;
       float deltaE=0;
       ++isc;
       reco::BasicClusterRefVector::iterator b_iter = it_super->clustersBegin();;
       for ( ; b_iter !=it_super->clustersEnd(); ++b_iter ) {  
          // Get strip position at the intersection point EE - Vertex: 	 
	 //const Point & point = b_iter->position();     	 
	  ESDetId strip1 = getClosestCellInPlane( &(*b_iter)->position(),1); // should be ported from Calorimetry/Preshower/interface/PreshBase.h in ORCA
	  ESDetId strip2 = getClosestCellInPlane( &(*b_iter)->position(),2);
        
          // Get a vector of ES clusters (found by the PreshSeeded algorithm) associated with a EE basic cluster. 
          // make the clusters by passing rechits to the agorithm
          for (int i=0; i<PreshNclust_; i++) {
	     reco::PreshowerCluster cl1 = presh_algo_->makeOneCluster(strip1, theCaloTopology);           
             clusters1.push_back(cl1);
	     reco::PreshowerCluster cl2 = presh_algo_->makeOneCluster(strip2, theCaloTopology);           
             clusters2.push_back(cl2);
	    
	     // save preshower clusters associated with a given BasicCluster
             &(*b_iter)->add_preshCl(cl1);  // should be added to BasicCluster.h
             &(*b_iter)->add_preshCl(cl2);
             
             e1 += cl1.Energy();
             e2 += cl2.Energy();
          }         
       }
       std::cout << " For SC #" << isc-1 << ", containing " << it_super->clustersSize() 
                 << " basic clusters, PreshowerClusterAlgo made " 
		 << clusters1.size() << " in X plane and " << clusters1.size() 
                 << " in Y plane " << " preshower clusters " << std::endl;

       // update energy of SuperCluster    
        if(e1+e2 > 1.0e-10)
           deltaE = miptogev_*(calib_plane1_*e1+calib_plane2_*e2);
       float uE = it_super->uncorrectedEnergy() + deltaE;
       float corrfact = it_super->Energy() / it_super->uncorrectedEnergy();
       float E = corrfact * uE;
       it_super->setUncorrectedEnergy(uE);   // setUncorrectedEnergy() and setEnergy() methods should exist
       it_super->setEnergy(E);
  }

   // create an auto_ptr to a PreshowerClusterCollection, copy the preshower clusters into it and put in the Event:
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p1(new reco::PreshowerClusterCollection);
   clusters_p1->assign(clusters1.begin(), clusters1.end());
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p2(new reco::PreshowerClusterCollection);
   clusters_p2->assign(clusters2.begin(), clusters2.end());

   // put collection of preshower clusters to the event
   evt.put( clusters_p1, clusterCollection1_ );
   evt.put( clusters_p2, clusterCollection2_ );

}

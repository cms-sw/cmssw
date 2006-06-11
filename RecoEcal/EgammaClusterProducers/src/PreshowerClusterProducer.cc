
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterProducer.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


#include "CLHEP/Geometry/Point3D.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
// #include "DataFormats/EgammaReco/interface/BasicCluster.h"
// #include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


PreshowerClusterProducer::PreshowerClusterProducer(const edm::ParameterSet& ps) {

  // use configuration file to setup input/output collection names 
  hitProducer_   = ps.getParameter<std::string>("ESRecHitProducer");
  hitCollection_ = ps.getParameter<std::string>("ESRecHitCollection");

  // Name of a SuperClusterCollection to make associations:
  SClusterCollection_ = ps.getParameter<std::string>("SuperClusterCollection");

  // calibration parameters:
  calib_plane1_ = ps.getUntrackedParameter<double>("PreshCalibPlane_X",1.0);
  calib_plane2_ = ps.getUntrackedParameter<double>("PreshCalibPlane_Y",0.70);
  miptogev_     = ps.getUntrackedParameter<double>("PreshCalibMIPtoGeV",0.024);

  // Output collections:
  clusterCollection1_ = ps.getParameter<std::string>("PreshClusterCollection_X");
  clusterCollection2_ = ps.getParameter<std::string>("PreshClusterCollection_Y");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  produces< reco::PreshowerClusterCollection >(clusterCollection1_);
  produces< reco::PreshowerClusterCollection >(clusterCollection2_);
  produces< reco::SuperClusterCollection >(superclusterCollection_);

 float PreshStripECut = ps.getUntrackedParameter<double>("PreshStripEnergyCut",0);
  float PreshClustECut = ps.getUntrackedParameter<double>("PreshClusterEnergyCut",0);
    int PreshSeededNst = ps.getUntrackedParameter<int>("PreshSeededNstrip",15);

  presh_algo = new PreshowerClusterAlgo(PreshStripECut,PreshClustECut,PreshSeededNst);

  PreshNclust_            = ps.getUntrackedParameter<int>("PreshNclust",4);

  nEvt_ = 0;  
}

PreshowerClusterProducer::~PreshowerClusterProducer() {
   delete presh_algo;
}


void PreshowerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  using namespace edm;

  Handle< EcalRecHitCollection >   pRecHits;
  Handle< reco::SuperClusterCollection > pSuperClusters;

  edm::ESHandle<CaloTopology> theCaloTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopology);     

  // get the ECAL geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);

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
  presh_algo->PreshHitsInit( *rechits );

  reco::PreshowerClusterCollection clusters1, clusters2;   // output collection of corrected PCs
  reco::SuperClusterCollection new_SC; // output collection of corrected SCs
   
  //make cycle over super clusters
  reco::SuperClusterCollection::const_iterator it_super;
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
	  Point point = (*b_iter)->position();  	 
	  ESDetId strip1 = getClosestCellInPlane( point, 1); 
	  ESDetId strip2 = getClosestCellInPlane( point, 2);
        
          // Get a vector of ES clusters (found by the PreshSeeded algorithm) associated with a EE basic cluster. 
          // make the clusters by passing rechits to the agorithm
          for (int i=0; i<PreshNclust_; i++) {
	     reco::PreshowerCluster cl1 = presh_algo->makeOneCluster(strip1, theCaloTopology, geoHandle);           
             clusters1.push_back(cl1);
	     reco::PreshowerCluster cl2 = presh_algo->makeOneCluster(strip2, theCaloTopology, geoHandle);           
             clusters2.push_back(cl2);
	    
// 	     // save preshower clusters associated with a given BasicCluster
//              &(*b_iter)->add_preshCl(cl1);  // should be added to BasicCluster.h
//              &(*b_iter)->add_preshCl(cl2);             
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
 //       float uE = it_super->uncorrectedEnergy() + deltaE;
//        float corrfact = it_super->energy() / it_super->uncorrectedEnergy();
//        float E = corrfact * uE;
       float E = it_super->energy() + deltaE;
       const reco::BasicClusterRefVector bc;
       //       bc.assign(it_super->clustersBegin(), it_super->clustersEnd());
       copy(it_super->clustersBegin(), it_super->clustersEnd(), bc.begin());
       reco::SuperCluster sc( E, it_super->position(), it_super->seed(), bc);
       new_SC.push_back(sc);
   }

   // create an auto_ptr to a PreshowerClusterCollection, copy the preshower clusters into it and put in the Event:
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p1(new reco::PreshowerClusterCollection);
   clusters_p1->assign(clusters1.begin(), clusters1.end());
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p2(new reco::PreshowerClusterCollection);
   clusters_p2->assign(clusters2.begin(), clusters2.end());

   // put collection of preshower clusters to the event
   evt.put( clusters_p1, clusterCollection1_ );
   evt.put( clusters_p2, clusterCollection2_ );
   std::cout << "Preshower clusters added to the event" << std::endl;

   // put new collection of corrected super clusters to the event

   std::cout << "   Found  " << new_SC.size() << " superclusters." << std::endl;  
   std::cout << "Initial#: " <<  SClusts->size() << " superclusters." << std::endl;  
   std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
   superclusters_p->assign(new_SC.begin(), new_SC.end());
   evt.put(superclusters_p, superclusterCollection_);

   std::cout << "Corrected SClusters added to the event" << std::endl;

   nEvt_++;
}

const ESDetId PreshowerClusterProducer::getClosestCellInPlane(Point &point, const int plane) const
{
  std::cout << "inside getClosestCellInPlane: x,y,z " << point.x() << std::endl;
    const ESDetId startES_1(2,15,2,1,1); 
    const ESDetId startES_2(2,15,2,2,1);
    if ( plane == 1 ) {
      std::cout << "getClosestCellInPlane: Starting at " << startES_1 << std::endl;        
      return startES_1;
    }
    else if ( plane == 2 ) {
      std::cout << "getClosestCellInPlane: Starting at " << startES_2 << std::endl;        
      return startES_2;
    }
    else { std::cout << "Wrong plane number" << std::endl; 
      return ESDetId(0);
   }
}

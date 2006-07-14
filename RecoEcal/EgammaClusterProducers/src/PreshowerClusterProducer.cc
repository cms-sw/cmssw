// system include files
#include <vector>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include <fstream>

#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterProducer.h"

#include "TFile.h"

///----

PreshowerClusterProducer::PreshowerClusterProducer(const edm::ParameterSet& ps) {

  // use configuration file to setup input/output collection names 
  preshHitProducer_   = ps.getParameter<std::string>("preshRecHitProducer");
  preshHitCollection_ = ps.getParameter<std::string>("preshRecHitCollection");

  // Name of a SuperClusterCollection to make associations:
  endcapSClusterCollection_ = ps.getParameter<std::string>("endcapSClusterCollection");
  endcapSClusterProducer_   = ps.getParameter<std::string>("endcapSClusterProducer");

  // Output collections:
  preshClusterCollectionX_ = ps.getParameter<std::string>("preshClusterCollectionX");
  preshClusterCollectionY_ = ps.getParameter<std::string>("preshClusterCollectionY");
  preshNclust_             = ps.getParameter<int>("preshNclust");

  // calibration parameters:
  calib_planeX_ = ps.getParameter<double>("preshCalibPlaneX");
  calib_planeY_ = ps.getParameter<double>("preshCalibPlaneY");
  miptogev_     = ps.getParameter<double>("preshCalibMIPtoGeV");

  assocSClusterCollection_ = ps.getParameter<std::string>("assocSClusterCollection");

  produces< reco::PreshowerClusterCollection >(preshClusterCollectionX_);
  produces< reco::PreshowerClusterCollection >(preshClusterCollectionY_);
  produces< reco::SuperClusterCollection >(assocSClusterCollection_);

  float preshStripECut = ps.getParameter<double>("preshStripEnergyCut");
  float preshClustECut = ps.getParameter<double>("preshClusterEnergyCut");
    int preshSeededNst = ps.getParameter<int>("preshSeededNstrip");

  // The debug level
  std::string debugString = ps.getParameter<std::string>("debugLevel");
  if      (debugString == "DEBUG")   debugL = pDEBUG;
  else if (debugString == "INFO")    debugL = pINFO;
  else                               debugL = pERROR;

  presh_algo = new PreshowerClusterAlgo(preshStripECut,preshClustECut,preshSeededNst,debugL);

  nEvt_ = 0;  

}



PreshowerClusterProducer::~PreshowerClusterProducer() {
   delete presh_algo;
}


void PreshowerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  if ( debugL <= pINFO ) std::cout << "\n .......  Event # " << nEvt_+1 << " is analyzing ....... " << std::endl << std::endl;
  
  edm::Handle< EcalRecHitCollection >   pRecHits;
  edm::Handle< reco::SuperClusterCollection > pSuperClusters;

  // get the ECAL geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);

//   const CaloSubdetectorGeometry *geometry_p;
//   geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;

  CaloSubdetectorTopology *topology = new EcalPreshowerTopology(geoHandle); 
  CaloSubdetectorTopology *& topology_p = topology;

 // fetch the product (pSuperClusters)
  try {
    evt.getByLabel(endcapSClusterProducer_, endcapSClusterCollection_, pSuperClusters);   
  } catch ( cms::Exception& ex ) {
    edm::LogError("PreshowerClusterProducerError ") << "Error! can't get the product " << endcapSClusterCollection_.c_str();
  }
  // pointer to the object in the product
  const reco::SuperClusterCollection* SClusts = pSuperClusters.product();
  if ( debugL <= pINFO ) std::cout <<"### Total # Endcap Superclusters: " << SClusts->size() << std::endl;

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = SClusts->begin(); aClus != SClusts->end(); aClus++) {
    if ( debugL == pDEBUG ) std::cout << "PreshowerClusterProducerInfo: superEenergy = " << aClus->energy() << std::endl;
  }

  // fetch the product (RecHits)
  try {
    evt.getByLabel( preshHitProducer_, preshHitCollection_, pRecHits);
  } catch ( cms::Exception& ex ) {
    edm::LogError("PreshowerClusterProducerError ") << "Error! can't get the product " << preshHitCollection_.c_str();
  }
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product(); // EcalRecHitCollection hit_collection = *rhcHandle;
  if ( debugL == pDEBUG ) std::cout << "PreshowerClusterProducerInfo: ### Total # of preshower RecHits: " << rechits->size() << std::endl;;

 
  // make the map of rechits:
  std::map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
     //Make the map of DetID, EcalRecHit pairs
     rechits_map.insert(std::make_pair(it->id(), *it));   
  }

  if ( debugL <= pINFO ) std::cout << "PreshowerClusterProducerInfo: ### rechits_map of size " 
                                         << rechits_map.size() <<" was created!" << std::endl;   

  reco::PreshowerClusterCollection clusters1, clusters2;   // output collection of corrected PCs
  reco::SuperClusterCollection new_SC; // output collection of corrected SCs
  reco::BasicClusterRefVector new_BC; // output collection of corrected SCs

  if ( debugL == pDEBUG ) std::cout << " Making a cycle over Superclusters ..." << std::endl; 
  //make cycle over super clusters
  reco::SuperClusterCollection::const_iterator it_super;
  int isc = 0;
  for ( it_super=SClusts->begin();  it_super!=SClusts->end(); it_super++ ) {     
       float e1=0;
       float e2=0;
       float deltaE=0;
       ++isc;

       if ( debugL <= pINFO ) std::cout << " superE = " << it_super->energy() << " superETA = " << it_super->eta() 
       		                                       << " superPHI = " << it_super->phi() << std::endl;
       if ( debugL == pINFO ) std::cout << " This SC contains " << it_super->clustersSize() << " BCs" << std::endl;
       reco::BasicClusterRefVector::iterator b_iter = it_super->clustersBegin();
       for ( ; b_iter !=it_super->clustersEnd(); ++b_iter ) {  

          // Get strip position at intersection point of the line EE - Vertex:
         double X = (*b_iter)->x();
	 double Y = (*b_iter)->y();
         double Z = (*b_iter)->z();
	 const GlobalPoint point(X,Y,Z);         

	 ESDetId strip1((dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 1)); 
         ESDetId strip2((dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 2));

         if ( debugL <= pINFO ) {
	    if ( strip1 != ESDetId(0) && strip2 != ESDetId(0) ) {
              std::cout << " Intersected preshower strips are: " << std::endl;
              std::cout << strip1 << std::endl;
	      std::cout << strip2 << std::endl;
	    }
            else if ( strip1 == ESDetId(0) )
              std::cout << " No intersected strip in plane 1 " << std::endl;
            else if ( strip2 == ESDetId(0) )
              std::cout << " No intersected strip in plane 2 " << std::endl;
         }

         // Get a vector of ES clusters (found by the PreshSeeded algorithm) associated with a given EE basic cluster.         
         for (int i=0; i<preshNclust_; i++) {
	     reco::PreshowerCluster cl1 = presh_algo->makeOneCluster(strip1,&rechits_map,b_iter,geometry_p,topology_p);           
             clusters1.push_back(cl1);
	     reco::PreshowerCluster cl2 = presh_algo->makeOneCluster(strip2,&rechits_map,b_iter,geometry_p,topology_p);           
             clusters2.push_back(cl2);
	                   
             e1 += cl1.energy();
             e2 += cl2.energy();

          } // end of cycle over ES clusters

             new_BC.push_back(*b_iter);

         }  // end of cycle over BCs

       if ( debugL <= pINFO ) std::cout << " For SC #" << isc-1 << ", containing " << it_super->clustersSize() 
                 << " basic clusters, PreshowerClusterAlgo made " 
		 << clusters1.size() << " in X plane and " << clusters2.size() 
                 << " in Y plane " << " preshower clusters " << std::endl;

       // update energy of the SuperCluster    
       if(e1+e2 > 1.0e-10)
           deltaE = miptogev_*(calib_planeX_*e1+calib_planeY_*e2);       

       float E = it_super->energy() + deltaE;
       
       if ( debugL == pDEBUG ) std::cout << " Creating corrected SC " << std::endl;
       reco::SuperCluster sc( E, it_super->position(), it_super->seed(), new_BC);
       new_SC.push_back(sc);
       if ( debugL <= pINFO ) std::cout << " SuperClusters energies: old E = " << sc.energy() 
                                        << " and new E =" << it_super->energy() << std::endl;

   } // end of cycle over SCs

   // create an auto_ptr to a PreshowerClusterCollection, copy the preshower clusters into it and put in the Event:
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p1(new reco::PreshowerClusterCollection);
   clusters_p1->assign(clusters1.begin(), clusters1.end());
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p2(new reco::PreshowerClusterCollection);
   clusters_p2->assign(clusters2.begin(), clusters2.end());
   // put collection of preshower clusters to the event
   evt.put( clusters_p1, preshClusterCollectionX_ );
   evt.put( clusters_p2, preshClusterCollectionY_ );
   if ( debugL <= pINFO ) std::cout << "Preshower clusters added to the event" << std::endl;

   // put new collection of corrected super clusters to the event
   std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
   superclusters_p->assign(new_SC.begin(), new_SC.end());
   evt.put(superclusters_p, assocSClusterCollection_);
   if ( debugL <= pINFO ) std::cout << "Corrected SClusters added to the event" << std::endl;

   nEvt_++;

}


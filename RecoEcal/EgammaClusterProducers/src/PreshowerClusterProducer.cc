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


using namespace edm;
using namespace std;


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
  gamma_        = ps.getParameter<double>("preshCalibGamma");
  mip_          = ps.getParameter<double>("preshCalibMIP");

  assocSClusterCollection_ = ps.getParameter<std::string>("assocSClusterCollection");

  produces< reco::PreshowerClusterCollection >(preshClusterCollectionX_);
  produces< reco::PreshowerClusterCollection >(preshClusterCollectionY_);
  produces< reco::SuperClusterCollection >(assocSClusterCollection_);

  float preshStripECut = ps.getParameter<double>("preshStripEnergyCut");
    int preshSeededNst = ps.getParameter<int>("preshSeededNstrip");
  preshClustECut = ps.getParameter<double>("preshClusterEnergyCut");

  // The debug level
  std::string debugString = ps.getParameter<std::string>("debugLevel");
  if      (debugString == "DEBUG")   debugL = PreshowerClusterAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL = PreshowerClusterAlgo::pINFO;
  else                               debugL = PreshowerClusterAlgo::pERROR;

  presh_algo = new PreshowerClusterAlgo(preshStripECut,preshClustECut,preshSeededNst,debugL);

  nEvt_ = 0;  

}


PreshowerClusterProducer::~PreshowerClusterProducer() {
   delete presh_algo;
}


void PreshowerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  edm::Handle< EcalRecHitCollection >   pRecHits;
  edm::Handle< reco::SuperClusterCollection > pSuperClusters;

  // get the ECAL geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;

  EcalPreshowerTopology topology(geoHandle);
  CaloSubdetectorTopology * topology_p = &topology;

 // fetch the product (pSuperClusters)
  evt.getByLabel(endcapSClusterProducer_, endcapSClusterCollection_, pSuperClusters);   
  const reco::SuperClusterCollection* SClusts = pSuperClusters.product();
  if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout <<"### Total # Endcap Superclusters: " << SClusts->size() << std::endl;

  // fetch the product (RecHits)
  evt.getByLabel( preshHitProducer_, preshHitCollection_, pRecHits);
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product(); // EcalRecHitCollection hit_collection = *rhcHandle;
  if ( debugL == PreshowerClusterAlgo::pDEBUG ) std::cout << "PreshowerClusterProducerInfo: ### Total # of preshower RecHits: " 
                                                          << rechits->size() << std::endl;
  if ( rechits->size() <= 0 ) return;

  // make the map of rechits:
  std::map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
     //Make the map of DetID, EcalRecHit pairs
     rechits_map.insert(std::make_pair(it->id(), *it));   
  }
  // The set of used DetID's for a given event:
  std::set<DetId> used_strips;
  used_strips.clear();

  if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout << "PreshowerClusterProducerInfo: ### rechits_map of size " 
                                         << rechits_map.size() <<" was created!" << std::endl;   

  reco::PreshowerClusterCollection clusters1, clusters2;   // output collection of corrected PCs
  reco::SuperClusterCollection new_SC; // output collection of corrected SCs
  reco::BasicClusterRefVector new_BC; // output collection of corrected BCs

  if ( debugL == PreshowerClusterAlgo::pDEBUG ) std::cout << " Making a cycle over Superclusters ..." << std::endl; 
  //make cycle over super clusters
  reco::SuperClusterCollection::const_iterator it_super;
  int isc = 0;
  for ( it_super=SClusts->begin();  it_super!=SClusts->end(); it_super++ ) {     
       float e1=0;
       float e2=0;
       float deltaE=0;
       ++isc;

       if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout << " superE = " << it_super->energy() << " superETA = " << it_super->eta() 
       		                                       << " superPHI = " << it_super->phi() << std::endl;
       if ( debugL == PreshowerClusterAlgo::pINFO ) std::cout << " This SC contains " << it_super->clustersSize() << " BCs" << std::endl;

       reco::BasicClusterRefVector::iterator bc_iter = it_super->clustersBegin();
       for ( ; bc_iter !=it_super->clustersEnd(); ++bc_iter ) {  

       // Get strip position at intersection point of the line EE - Vertex:
         double X = (*bc_iter)->x();
	 double Y = (*bc_iter)->y();
         double Z = (*bc_iter)->z();        
	 const GlobalPoint point(X,Y,Z);         

         DetId tmp1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 1);
         DetId tmp2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 2);
	 ESDetId strip1 = (tmp1 == DetId(0)) ? ESDetId(0) : ESDetId(tmp1);
	 ESDetId strip2 = (tmp2 == DetId(0)) ? ESDetId(0) : ESDetId(tmp2);     

         if ( debugL <= PreshowerClusterAlgo::pINFO ) {
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
	   reco::PreshowerCluster cl1 = presh_algo->makeOneCluster(strip1,&used_strips,&rechits_map,bc_iter,geometry_p,topology_p);   
             if ( cl1.energy() > preshClustECut) {
               clusters1.push_back(cl1);
               e1 += cl1.energy();       
             }
	     reco::PreshowerCluster cl2 = presh_algo->makeOneCluster(strip2,&used_strips,&rechits_map,bc_iter,geometry_p,topology_p); 

             if ( cl2.energy() > preshClustECut) {
               clusters2.push_back(cl2);
               e2 += cl2.energy();
             }	                               

          } // end of cycle over ES clusters

            new_BC.push_back(*bc_iter);

         }  // end of cycle over BCs

       if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout << " For SC #" << isc-1 << ", containing " << it_super->clustersSize() 
                 << " basic clusters, PreshowerClusterAlgo made " 
		 << clusters1.size() << " in X plane and " << clusters2.size() 
                 << " in Y plane " << " preshower clusters " << std::endl;

       // update energy of the SuperCluster    
       if(e1+e2 > 1.0e-10) {
	 // GeV to #MIPs
	   e1 = e1 / mip_;
           e2 = e2 / mip_;
           deltaE = gamma_*(calib_planeX_*e1+calib_planeY_*e2);       
       }

       //corrected Energy
       float E = it_super->energy() + deltaE;
       
       if ( debugL == PreshowerClusterAlgo::pDEBUG ) std::cout << " Creating corrected SC " << std::endl;
       reco::SuperCluster sc( E, it_super->position(), it_super->seed(), new_BC);
       new_SC.push_back(sc);
       if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout << " SuperClusters energies: new E = " << sc.energy() 
                                        << " vs. old E =" << it_super->energy() << std::endl;

   } // end of cycle over SCs
  

   // create an auto_ptr to a PreshowerClusterCollection, copy the preshower clusters into it and put in the Event:
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p1(new reco::PreshowerClusterCollection);
   clusters_p1->assign(clusters1.begin(), clusters1.end());
   std::auto_ptr< reco::PreshowerClusterCollection > clusters_p2(new reco::PreshowerClusterCollection);
   clusters_p2->assign(clusters2.begin(), clusters2.end());
   // put collection of preshower clusters to the event
   evt.put( clusters_p1, preshClusterCollectionX_ );
   evt.put( clusters_p2, preshClusterCollectionY_ );
   if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout << "Preshower clusters added to the event" << std::endl;

   // put new collection of corrected super clusters to the event
   std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
   superclusters_p->assign(new_SC.begin(), new_SC.end());
   evt.put(superclusters_p, assocSClusterCollection_);
   if ( debugL <= PreshowerClusterAlgo::pINFO ) std::cout << "Corrected SClusters added to the event" << std::endl;

   nEvt_++;

}

 

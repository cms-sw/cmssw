// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"


// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Class header file
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHybridClusterProducer.h"


EgammaHLTHybridClusterProducer::EgammaHLTHybridClusterProducer(const edm::ParameterSet& ps)
{

    // The debug level
  std::string debugString = ps.getParameter<std::string>("debugLevel");
  if      (debugString == "DEBUG")   debugL = HybridClusterAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL = HybridClusterAlgo::pINFO;
  else                               debugL = HybridClusterAlgo::pERROR;

  basicclusterCollection_ = ps.getParameter<std::string>("basicclusterCollection");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  hitproducer_ = ps.getParameter<edm::InputTag>("ecalhitproducer");
  hitcollection_ =ps.getParameter<std::string>("ecalhitcollection");



  // L1 matching parameters
  l1TagIsolated_ = ps.getParameter< edm::InputTag > ("l1TagIsolated");
  l1TagNonIsolated_ = ps.getParameter< edm::InputTag > ("l1TagNonIsolated");

  doIsolated_   = ps.getParameter<bool>("doIsolated");

  l1LowerThr_ = ps.getParameter<double> ("l1LowerThr");
  l1UpperThr_ = ps.getParameter<double> ("l1UpperThr");
  l1LowerThrIgnoreIsolation_ = ps.getParameter<double> ("l1LowerThrIgnoreIsolation");

  regionEtaMargin_   = ps.getParameter<double>("regionEtaMargin");
  regionPhiMargin_   = ps.getParameter<double>("regionPhiMargin");

  // Parameters for the position calculation:
  posCalculator_ = PositionCalc( ps.getParameter<edm::ParameterSet>("posCalcParameters") );
  

  hybrid_p = new HybridClusterAlgo(ps.getParameter<double>("HybridBarrelSeedThr"), 
                                   ps.getParameter<int>("step"),
                                   ps.getParameter<double>("ethresh"),
                                   ps.getParameter<double>("eseed"),
                                   ps.getParameter<double>("ewing"),
                                   ps.getParameter<std::vector<int> >("RecHitFlagToBeExcluded"),
                                   posCalculator_,
                                   debugL,
			           ps.getParameter<bool>("dynamicEThresh"),
                                   ps.getParameter<double>("eThreshA"),
                                   ps.getParameter<double>("eThreshB"),
				   ps.getParameter<std::vector<int> >("RecHitSeverityToBeExcluded"),
				   ps.getParameter<double>("severityRecHitThreshold"),
				   ps.getParameter<int>("severitySpikeId"),
				   ps.getParameter<double>("severitySpikeThreshold"),
				   ps.getParameter<bool>("excludeFlagged")
				   );

  bool dynamicPhiRoad = ps.getParameter<bool>("dynamicPhiRoad");
    if (dynamicPhiRoad) {
     edm::ParameterSet bremRecoveryPset = ps.getParameter<edm::ParameterSet>("bremRecoveryPset");
     hybrid_p->setDynamicPhiRoad(bremRecoveryPset);
  }


  produces< reco::BasicClusterCollection >(basicclusterCollection_);
  produces< reco::SuperClusterCollection >(superclusterCollection_);
  nEvt_ = 0;
}


EgammaHLTHybridClusterProducer::~EgammaHLTHybridClusterProducer()
{
  delete hybrid_p;
}


void EgammaHLTHybridClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  //  evt.getByType(rhcHandle);
  evt.getByLabel(hitproducer_.label(), hitcollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  const EcalRecHitCollection *hit_collection = rhcHandle.product();

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;
  std::auto_ptr<const CaloSubdetectorTopology> topology;

  edm::ESHandle<EcalChannelStatus> chStatus;
  es.get<EcalChannelStatusRcd>().get(chStatus);
  const EcalChannelStatus* theEcalChStatus = chStatus.product();

  //if (debugL == HybridClusterAlgo::pDEBUG)
  //std::cout << "\n\n\n" << hitcollection_ << "\n\n" << std::endl;

  if(hitcollection_ == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    topology.reset(new EcalBarrelTopology(geoHandle));
  } else if(hitcollection_ == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    topology.reset(new EcalEndcapTopology(geoHandle));
  } else if(hitcollection_ == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    topology.reset(new EcalPreshowerTopology (geoHandle));
  } else throw(std::runtime_error("\n\nHybrid Cluster Producer encountered invalied ecalhitcollection type.\n\n"));
    
  //Get the L1 EM Particle Collection
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  if(doIsolated_)
    evt.getByLabel(l1TagIsolated_, emIsolColl);
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  evt.getByLabel(l1TagNonIsolated_, emNonIsolColl);

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  es.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  std::vector<EcalEtaPhiRegion> regions;

  if(doIsolated_) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emIsolColl->begin(); emItr != emIsolColl->end() ;++emItr ){

    if (emItr->et() > l1LowerThr_ && emItr->et() < l1UpperThr_
        //&&
	//!emItr->gctEmCand()->regionId().isForward()
) {

      //bool isolated = emItr->gctEmCand()->isolated();
      //if ((l1Isolated_ &&isolated) || (!l1Isolated_ &&!isolated)) {

      // Access the GCT hardware object corresponding to the L1Extra EM object.
      int etaIndex = emItr->gctEmCand()->etaIndex() ;
      int phiIndex = emItr->gctEmCand()->phiIndex() ;
      // Use the L1CaloGeometry to find the eta, phi bin boundaries.
      double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
      double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
      double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
      double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

       int isbarl=0;
       //Part of the region is in the barel if either the upper or lower
       //edge of the region is within the barrel
       if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	  ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;

	//std::cout<<"Hybrid etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;

      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;

      if (etaHigh>1.479) etaHigh=1.479;
      if (etaLow<-1.479) etaLow=-1.479;

      if(isbarl) regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));

    }
  }
  }

  if(!doIsolated_||l1LowerThrIgnoreIsolation_<64) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emNonIsolColl->begin(); emItr != emNonIsolColl->end() ;++emItr ){

      if(doIsolated_&&emItr->et()<l1LowerThrIgnoreIsolation_) continue;

    if (emItr->et() > l1LowerThr_ && emItr->et() < l1UpperThr_
        //&&
	//!emItr->gctEmCand()->regionId().isForward()
) {

      //bool isolated = emItr->gctEmCand()->isolated();
      //if ((l1Isolated_ &&isolated) || (!l1Isolated_ &&!isolated)) {

      // Access the GCT hardware object corresponding to the L1Extra EM object.
      int etaIndex = emItr->gctEmCand()->etaIndex() ;
      int phiIndex = emItr->gctEmCand()->phiIndex() ;
      // Use the L1CaloGeometry to find the eta, phi bin boundaries.
      double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
      double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
      double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
      double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

       int isbarl=0;
       //Part of the region is in the barel if either the upper or lower
       //edge of the region is within the barrel
       if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	  ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;

	//std::cout<<"Hybrid etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;

      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;

      if (etaHigh>1.479) etaHigh=1.479;
      if (etaLow<-1.479) etaLow=-1.479;

      if(isbarl) regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));

    }
  }
  }

  // make the Basic clusters!
  reco::BasicClusterCollection basicClusters;
  hybrid_p->makeClusters(hit_collection, geometry_p, basicClusters, true, regions,theEcalChStatus);
  //if (debugL == HybridClusterAlgo::pDEBUG)
  //std::cout << "Hybrid Finished clustering - BasicClusterCollection returned to producer..." << std::endl;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > basicclusters_p(new reco::BasicClusterCollection);
  basicclusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  evt.put(basicclusters_p, 
                                                                       basicclusterCollection_);
  //Basic clusters now in the event.
  //if (debugL == HybridClusterAlgo::pDEBUG)
  //std::cout << "Basic Clusters now put into event." << std::endl;
  
  //Weird though it is, get the BasicClusters back out of the event.  We need the
  //edm::Ref to these guys to make our superclusters for Hybrid.
//  edm::Handle<reco::BasicClusterCollection> bccHandle;
 // evt.getByLabel("clusterproducer",basicclusterCollection_, bccHandle);
  if (!(bccHandle.isValid())) {
    //if (debugL <= HybridClusterAlgo::pINFO)
    //std::cout << "could not get a handle on the BasicClusterCollection!" << std::endl;
    return;
  }
  reco::BasicClusterCollection clusterCollection = *bccHandle;
  //if (debugL == HybridClusterAlgo::pDEBUG)
  //std::cout << "Got the BasicClusterCollection" << std::endl;

  reco::CaloClusterPtrVector clusterRefVector;
  for (unsigned int i = 0; i < clusterCollection.size(); i++){
    clusterRefVector.push_back(reco::CaloClusterPtr(bccHandle, i));
  }

  reco::SuperClusterCollection superClusters = hybrid_p->makeSuperClusters(clusterRefVector);
  //if (debugL == HybridClusterAlgo::pDEBUG)
  //std::cout << "Found: " << superClusters.size() << " superclusters." << std::endl;

  std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
  superclusters_p->assign(superClusters.begin(), superClusters.end());
  evt.put(superclusters_p, superclusterCollection_);

  //if (debugL == HybridClusterAlgo::pDEBUG)
  //std::cout << "Hybrid Clusters (Basic/Super) added to the Event! :-)" << std::endl;

  nEvt_++;
}


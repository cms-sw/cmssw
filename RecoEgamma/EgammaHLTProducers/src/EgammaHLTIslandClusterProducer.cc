// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// Level 1 Trigger
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

// Class header file
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTIslandClusterProducer.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

EgammaHLTIslandClusterProducer::EgammaHLTIslandClusterProducer(const edm::ParameterSet& ps)
  : doBarrel_   (ps.getParameter<bool>("doBarrel"))
  , doEndcaps_  (ps.getParameter<bool>("doEndcaps"))
  , doIsolated_ (ps.getParameter<bool>("doIsolated"))

  // Parameters to identify the hit collections
  , barrelHitCollection_   (ps.getParameter<edm::InputTag>("barrelHitProducer"))
  , endcapHitCollection_   (ps.getParameter<edm::InputTag>("endcapHitProducer"))
  , barrelHitToken_   (consumes<EcalRecHitCollection>(barrelHitCollection_))
  , endcapHitToken_   (consumes<EcalRecHitCollection>(endcapHitCollection_))

  // The names of the produced cluster collections
  , barrelClusterCollection_  (ps.getParameter<std::string>("barrelClusterCollection"))
  , endcapClusterCollection_  (ps.getParameter<std::string>("endcapClusterCollection"))

  // L1 matching parameters
  , l1TagIsolated_    (consumes<l1extra::L1EmParticleCollection>(ps.getParameter< edm::InputTag > ("l1TagIsolated")))
  , l1TagNonIsolated_ (consumes<l1extra::L1EmParticleCollection>(ps.getParameter< edm::InputTag > ("l1TagNonIsolated")))
  , l1LowerThr_ (ps.getParameter<double> ("l1LowerThr"))
  , l1UpperThr_ (ps.getParameter<double> ("l1UpperThr"))
  , l1LowerThrIgnoreIsolation_ (ps.getParameter<double> ("l1LowerThrIgnoreIsolation"))

  , regionEtaMargin_   (ps.getParameter<double>("regionEtaMargin"))
  , regionPhiMargin_   (ps.getParameter<double>("regionPhiMargin"))

   // Parameters for the position calculation:
  , posCalculator_ (PositionCalc( ps.getParameter<edm::ParameterSet>("posCalcParameters") ))
  // Island algorithm parameters
  , verb_ (ps.getParameter<std::string>("VerbosityLevel"))
  , island_p (new IslandClusterAlgo(ps.getParameter<double>("IslandBarrelSeedThr"),
                                     ps.getParameter<double>("IslandEndcapSeedThr"),
                                     posCalculator_,
                                     StringToEnumValue<EcalRecHit::Flags>(ps.getParameter<std::vector<std::string> >("SeedRecHitFlagToBeExcludedEB")),
                                     StringToEnumValue<EcalRecHit::Flags>(ps.getParameter<std::vector<std::string> >("SeedRecHitFlagToBeExcludedEE")),
                                     StringToEnumValue<EcalRecHit::Flags>(ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEB")),
                                     StringToEnumValue<EcalRecHit::Flags>(ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcludedEE")),
                                     verb_ == "DEBUG" ? IslandClusterAlgo::pDEBUG :
                                     (verb_ == "WARNING" ? IslandClusterAlgo::pWARNING :
                                      (verb_ == "INFO" ? IslandClusterAlgo::pINFO :
                                       IslandClusterAlgo::pERROR)))
                                     )
{
  // Produces a collection of barrel and a collection of endcap clusters
  produces< reco::BasicClusterCollection >(endcapClusterCollection_);
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);
}

EgammaHLTIslandClusterProducer::~EgammaHLTIslandClusterProducer() {
  delete island_p;
}

void EgammaHLTIslandClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<std::string>("VerbosityLevel", "ERROR");
  desc.add<bool>("doBarrel", true);
  desc.add<bool>("doEndcaps", true);
  desc.add<bool>("doIsolated", true);
  desc.add<edm::InputTag>("barrelHitProducer", edm::InputTag("islandEndcapBasicClusters", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapHitProducer", edm::InputTag("islandEndcapBasicClusters", "EcalRecHitsEB"));
  desc.add<std::string>("barrelClusterCollection", "islandBarrelBasicClusters");
  desc.add<std::string>("endcapClusterCollection", "islandEndcapBasicClusters");
  desc.add<double>("IslandBarrelSeedThr", 0.5);
  desc.add<double>("IslandEndcapSeedThr", 0.18);
  desc.add<edm::InputTag>("l1TagIsolated", edm::InputTag("l1extraParticles","Isolated"));
  desc.add<edm::InputTag>("l1TagNonIsolated", edm::InputTag("l1extraParticles","NonIsolated"));
  desc.add<double>("l1LowerThr", 0.0);
  desc.add<double>("l1UpperThr", 9999.0);
  desc.add<double>("l1LowerThrIgnoreIsolation", 9999.0);
  desc.add<double>("regionEtaMargin", 0.3);
  desc.add<double>("regionPhiMargin", 0.4);
  //desc.add<edm::ParameterSet>("posCalcParameters"), edm::ParameterSet());
  desc.add<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEE", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEE", {});
  descriptions.add("hltEgammaHLTIslandClusterProducer", desc);  
}

void EgammaHLTIslandClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  if(doIsolated_)
    evt.getByToken(l1TagIsolated_, emIsolColl);
  
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  evt.getByToken(l1TagNonIsolated_, emNonIsolColl);

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  es.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  std::vector<RectangularEtaPhiRegion> barrelRegions;
  std::vector<RectangularEtaPhiRegion> endcapRegions;

  if(doIsolated_) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emIsolColl->begin(); emItr != emIsolColl->end() ;++emItr ){

      if (emItr->et() > l1LowerThr_ && emItr->et() < l1UpperThr_) {
	
	// Access the GCT hardware object corresponding to the L1Extra EM object.
	int etaIndex = emItr->gctEmCand()->etaIndex() ;
	
	int phiIndex = emItr->gctEmCand()->phiIndex() ;
	// Use the L1CaloGeometry to find the eta, phi bin boundaries.
	double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
	double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
	double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
	double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

	//Attention isForward does not work
	int isforw=0;
	int isbarl=0;
	if((float)(etaHigh)>1.479 || (float)(etaLow)<-1.479) isforw=1;
	if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	   ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;

	//std::cout<<"Island etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;
	
	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	//if (emItr->gctEmCand()->regionId().isForward()) {
	if (isforw) {
	  if (etaHigh>-1.479 && etaHigh<1.479) etaHigh=-1.479;
	  if ( etaLow>-1.479 &&  etaLow<1.479) etaLow=1.479;
	  RectangularEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  endcapRegions.push_back(region);
	}
	if (isbarl) {
	  if (etaHigh>1.479) etaHigh=1.479;
	  if (etaLow<-1.479) etaLow=-1.479;
	  RectangularEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  barrelRegions.push_back(region);
	}
	RectangularEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	
      }
    }
  }


  if(!doIsolated_||l1LowerThrIgnoreIsolation_<64) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emNonIsolColl->begin(); emItr != emNonIsolColl->end() ;++emItr ){

      if(doIsolated_&&emItr->et()<l1LowerThrIgnoreIsolation_) continue;

      if (emItr->et() > l1LowerThr_ && emItr->et() < l1UpperThr_) {
	
	// Access the GCT hardware object corresponding to the L1Extra EM object.
	int etaIndex = emItr->gctEmCand()->etaIndex() ;
	
	
	int phiIndex = emItr->gctEmCand()->phiIndex() ;
	// Use the L1CaloGeometry to find the eta, phi bin boundaries.
	double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
	double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
	double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
	double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;


	int isforw=0;
	int isbarl=0;
	if((float)(etaHigh)>1.479 || (float)(etaLow)<-1.479) isforw=1;
	if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	   ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;

	//std::cout<<"Island etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;
	
	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	//if (emItr->gctEmCand()->regionId().isForward()) {
	if (isforw) {
	  if (etaHigh>-1.479 && etaHigh<1.479) etaHigh=-1.479;
	  if ( etaLow>-1.479 &&  etaLow<1.479) etaLow=1.479;
	  RectangularEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  endcapRegions.push_back(region);
	}
	if (isbarl) {
	  if (etaHigh>1.479) etaHigh=1.479;
	  if (etaLow<-1.479) etaLow=-1.479;
	  RectangularEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  barrelRegions.push_back(region);
	}
	
      }
    }
  }

  if (doEndcaps_ 
      //&&endcapRegions.size()!=0
      ) {

    clusterizeECALPart(evt, es, endcapHitToken_, endcapClusterCollection_, endcapRegions, IslandClusterAlgo::endcap);
  }
  if (doBarrel_ 
      //&& barrelRegions.size()!=0
      ) {
    clusterizeECALPart(evt, es, barrelHitToken_, barrelClusterCollection_, barrelRegions, IslandClusterAlgo::barrel);
  }
}


const EcalRecHitCollection * EgammaHLTIslandClusterProducer::getCollection(edm::Event& evt,
									   const edm::EDGetTokenT<EcalRecHitCollection>& hitToken_)
const {
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(hitToken_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      edm::LogError("EgammaHLTIslandClusterProducerError") << "Error! can't get the product ";
      return nullptr;
    } 
  return rhcHandle.product();
}


void EgammaHLTIslandClusterProducer::clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
							const edm::EDGetTokenT<EcalRecHitCollection>& hitToken,
							const std::string& clusterCollection,
							const std::vector<RectangularEtaPhiRegion>& regions,
							const IslandClusterAlgo::EcalPart& ecalPart)
const {
  // get the hit collection from the event:
  const EcalRecHitCollection *hitCollection_p = getCollection(evt, hitToken);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  if (ecalPart == IslandClusterAlgo::barrel) 
    {
      geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
      topology_p = new EcalBarrelTopology(geoHandle);
    }
  else
    {
      geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
      topology_p = new EcalEndcapTopology(geoHandle); 
   }

  const CaloSubdetectorGeometry *geometryES_p;
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  // Run the clusterization algorithm:
  reco::BasicClusterCollection clusters;
  clusters = island_p->makeClusters(hitCollection_p, geometry_p, topology_p, geometryES_p, ecalPart, true, regions);

  // create an unique_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  auto clusters_p = std::make_unique<reco::BasicClusterCollection>();
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  if (ecalPart == IslandClusterAlgo::barrel) 
    bccHandle = evt.put(std::move(clusters_p), barrelClusterCollection_);
  else
    bccHandle = evt.put(std::move(clusters_p), endcapClusterCollection_);

  delete topology_p;
}

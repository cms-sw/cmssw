// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

#include "CommonTools/Utils/interface/StringToEnumValue.h"
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
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Class header file
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTMulti5x5ClusterProducer.h"

EgammaHLTMulti5x5ClusterProducer::EgammaHLTMulti5x5ClusterProducer(const edm::ParameterSet& ps) {
  
  doBarrel_   = ps.getParameter<bool>("doBarrel");
  doEndcaps_  = ps.getParameter<bool>("doEndcaps");
  doIsolated_ = ps.getParameter<bool>("doIsolated");
  
  // Parameters to identify the hit collections
  barrelHitCollection_   = ps.getParameter<edm::InputTag>("barrelHitProducer");
  endcapHitCollection_   = ps.getParameter<edm::InputTag>("endcapHitProducer");
  barrelHitToken_ = consumes<EcalRecHitCollection>( barrelHitCollection_);
  endcapHitToken_ = consumes<EcalRecHitCollection>( endcapHitCollection_);

  // The names of the produced cluster collections
  barrelClusterCollection_  = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_  = ps.getParameter<std::string>("endcapClusterCollection");

  // Multi5x5 algorithm parameters
  double barrelSeedThreshold = ps.getParameter<double>("Multi5x5BarrelSeedThr");
  double endcapSeedThreshold = ps.getParameter<double>("Multi5x5EndcapSeedThr");

  // L1 matching parameters
  l1TagIsolated_    = consumes<l1extra::L1EmParticleCollection>(ps.getParameter< edm::InputTag > ("l1TagIsolated"));
  l1TagNonIsolated_ = consumes<l1extra::L1EmParticleCollection>(ps.getParameter< edm::InputTag > ("l1TagNonIsolated"));
  l1LowerThr_ = ps.getParameter<double> ("l1LowerThr");
  l1UpperThr_ = ps.getParameter<double> ("l1UpperThr");
  l1LowerThrIgnoreIsolation_ = ps.getParameter<double> ("l1LowerThrIgnoreIsolation");

  regionEtaMargin_   = ps.getParameter<double>("regionEtaMargin");
  regionPhiMargin_   = ps.getParameter<double>("regionPhiMargin");

  // Parameters for the position calculation:
  posCalculator_ = PositionCalc( ps.getParameter<edm::ParameterSet>("posCalcParameters") );

  const std::vector<std::string> flagnames = 
    ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");

  // exclude recHit flags from seeding
  std::vector<int> v_chstatus = StringToEnumValue<EcalRecHit::Flags>(flagnames);
  
  // Produces a collection of barrel and a collection of endcap clusters
  produces< reco::BasicClusterCollection >(endcapClusterCollection_);
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);

  Multi5x5_p = new Multi5x5ClusterAlgo(barrelSeedThreshold, endcapSeedThreshold, v_chstatus, posCalculator_);
  nEvt_ = 0;
}

EgammaHLTMulti5x5ClusterProducer::~EgammaHLTMulti5x5ClusterProducer() {
  delete Multi5x5_p;
}

void EgammaHLTMulti5x5ClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<bool>(("doBarrel"), false);
  desc.add<bool>(("doEndcaps"), true);
  desc.add<bool>(("doIsolated"), true);
  desc.add<std::string>("VerbosityLevel" ,"ERROR");

  edm::ParameterSetDescription posCalcPSET;
  posCalcPSET.add<double>("T0_barl", 7.4);
  posCalcPSET.add<double>("T0_endc", 3.1);
  posCalcPSET.add<double>("T0_endcPresh", 1.2);
  posCalcPSET.add<double>("W0", 4.2);
  posCalcPSET.add<double>("X0", 0.89);
  posCalcPSET.add<bool>("LogWeighted", true);
  desc.add<edm::ParameterSetDescription>("posCalcParameters", posCalcPSET);

  desc.add<edm::InputTag>(("barrelHitProducer"), edm::InputTag("hltEcalRegionalEgammaRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>(("endcapHitProducer"), edm::InputTag("hltEcalRegionalEgammaRecHit", "EcalRecHitsEE"));
  desc.add<std::string>(("barrelClusterCollection"), "notused");
  desc.add<std::string>(("endcapClusterCollection"), "multi5x5EndcapBasicClusters");
  desc.add<double>(("Multi5x5BarrelSeedThr"), 0.5);
  desc.add<double>(("Multi5x5EndcapSeedThr"), 0.5);
  desc.add<edm::InputTag>(("l1TagIsolated"), edm::InputTag("hltL1extraParticles","Isolated"));
  desc.add<edm::InputTag>(("l1TagNonIsolated"), edm::InputTag("hltL1extraParticles","NonIsolated"));
  desc.add<double>(("l1LowerThr"), 5.0);
  desc.add<double>(("l1UpperThr"), 9999.);
  desc.add<double>(("l1LowerThrIgnoreIsolation"), 999.0);
  desc.add<double>(("regionEtaMargin"), 0.3);
  desc.add<double>(("regionPhiMargin"), 0.4);

  desc.add<std::vector<std::string> >(("RecHitFlagToBeExcluded"), std::vector<std::string>());
  descriptions.add(("hltEgammaHLTMulti5x5ClusterProducer"), desc);  
}

void EgammaHLTMulti5x5ClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

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

  std::vector<EcalEtaPhiRegion> barrelRegions;
  std::vector<EcalEtaPhiRegion> endcapRegions;

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

	//std::cout<<"Multi5x5 etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;
	
	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	//if (emItr->gctEmCand()->regionId().isForward()) {
	if (isforw) {
	  if (etaHigh>-1.479 && etaHigh<1.479) etaHigh=-1.479;
	  if ( etaLow>-1.479 &&  etaLow<1.479) etaLow=1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  endcapRegions.push_back(region);
	}
	if (isbarl) {
	  if (etaHigh>1.479) etaHigh=1.479;
	  if (etaLow<-1.479) etaLow=-1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  barrelRegions.push_back(region);
	}
	EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	
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

	//std::cout<<"Multi5x5 etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;
	
	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	//if (emItr->gctEmCand()->regionId().isForward()) {
	if (isforw) {
	  if (etaHigh>-1.479 && etaHigh<1.479) etaHigh=-1.479;
	  if ( etaLow>-1.479 &&  etaLow<1.479) etaLow=1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  endcapRegions.push_back(region);
	}
	if (isbarl) {
	  if (etaHigh>1.479) etaHigh=1.479;
	  if (etaLow<-1.479) etaLow=-1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  barrelRegions.push_back(region);
	}
	
      }
    }
  }


  if (doEndcaps_) {
    clusterizeECALPart(evt, es, endcapHitToken_, endcapClusterCollection_, endcapRegions, reco::CaloID::DET_ECAL_ENDCAP);
  }
  if (doBarrel_) {
    clusterizeECALPart(evt, es, barrelHitToken_, barrelClusterCollection_, barrelRegions, reco::CaloID::DET_ECAL_BARREL);
  }
  nEvt_++;
}


const EcalRecHitCollection * EgammaHLTMulti5x5ClusterProducer::getCollection(edm::Event& evt,
									     edm::EDGetTokenT<EcalRecHitCollection>& hitToken) {

  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(hitToken, rhcHandle);
  
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      edm::LogError("EgammaHLTMulti5x5ClusterProducerError") << "Error! can't get the product ";
      return 0;
    } 
  return rhcHandle.product();
}


void EgammaHLTMulti5x5ClusterProducer::clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
							  edm::EDGetTokenT<EcalRecHitCollection>& hitToken,
							  const std::string& clusterCollection,
							  const std::vector<EcalEtaPhiRegion>& regions,
							  const reco::CaloID::Detectors detector) {

  // get the hit collection from the event:
  const EcalRecHitCollection *hitCollection_p = getCollection(evt, hitToken);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  if (detector == reco::CaloID::DET_ECAL_BARREL) 
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
  clusters = Multi5x5_p->makeClusters(hitCollection_p, geometry_p, topology_p, geometryES_p, detector, true, regions);

  // create an auto_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  if (detector == reco::CaloID::DET_ECAL_BARREL) 
    bccHandle = evt.put(clusters_p, barrelClusterCollection_);
  else
    bccHandle = evt.put(clusters_p, endcapClusterCollection_);

  delete topology_p;
}

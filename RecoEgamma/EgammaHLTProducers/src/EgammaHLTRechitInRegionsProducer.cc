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
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "DataFormats/EgammaReco/interface/BasicCluster.h"
//#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

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
//#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Class header file
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRechitInRegionsProducer.h"

//#include "DataFormats/Math/interface/deltaR.h"


EgammaHLTRechitInRegionsProducer::EgammaHLTRechitInRegionsProducer(const edm::ParameterSet& ps) {

  useUncalib_    = ps.getParameter<bool>("useUncalib");
  //hitproducer_   = ps.getParameter<edm::InputTag>("ecalhitproducer");

  l1TagIsolated_ = ps.getParameter< edm::InputTag > ("l1TagIsolated");
  l1TagNonIsolated_ = ps.getParameter< edm::InputTag > ("l1TagNonIsolated");
  doIsolated_   = ps.getParameter<bool>("doIsolated");
  
  l1LowerThr_ = ps.getParameter<double> ("l1LowerThr");
  l1UpperThr_ = ps.getParameter<double> ("l1UpperThr");
  l1LowerThrIgnoreIsolation_ = ps.getParameter<double> ("l1LowerThrIgnoreIsolation");

  regionEtaMargin_   = ps.getParameter<double>("regionEtaMargin");
  regionPhiMargin_   = ps.getParameter<double>("regionPhiMargin");

  //const std::vector<std::string> flagnames = ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");
  //const std::vector<int> flagsexcl = StringToEnumValue<EcalRecHit::Flags>(flagnames);
  
  //const std::vector<std::string> severitynames = ps.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcluded");
  //const std::vector<int> severitiesexcl = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  hitLabels     = ps.getParameter<std::vector<edm::InputTag>>("ecalhitLabels");
  productLabels = ps.getParameter<std::vector<std::string>>("productLabels");

  if (useUncalib_) {
    for (unsigned int i=0; i<hitLabels.size(); i++) { 
      uncalibHitTokens.push_back(consumes<EcalUncalibratedRecHitCollection>(hitLabels[i]));
      produces<EcalUncalibratedRecHitCollection>(productLabels[i]);
    }
  } else {
    for (unsigned int i=0; i<hitLabels.size(); i++) { 
      hitTokens.push_back(consumes<EcalRecHitCollection>(hitLabels[i]));
      produces<EcalRecHitCollection> (productLabels[i]);
    }
  }
}


EgammaHLTRechitInRegionsProducer::~EgammaHLTRechitInRegionsProducer()
{}

void EgammaHLTRechitInRegionsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> productTags;
  productTags.push_back("EcalRegionalRecHitsEB");
  productTags.push_back("EcalRegionalRecHitsEE");
  desc.add<std::vector<std::string>>("productLabels", productTags);
  std::vector<edm::InputTag> inputTags;
  inputTags.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEB"));
  inputTags.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEE"));
  inputTags.push_back(edm::InputTag("hltESRegionalEgammaRecHit:EcalRecHitsES"));
  desc.add<std::vector<edm::InputTag>>("ecalhitLabels", inputTags);
  //desc.add<edm::InputTag>("ecalhitproducer", edm::InputTag("ecalRecHit"));
  desc.add<edm::InputTag>("l1TagIsolated", edm::InputTag("l1extraParticles","Isolated"));
  desc.add<edm::InputTag>("l1TagNonIsolated", edm::InputTag("l1extraParticles","NonIsolated"));
  desc.add<bool>("useUncalib", true);
  desc.add<bool>("doIsolated", true);
  desc.add<double>("l1LowerThr", 5.0);
  desc.add<double>("l1UpperThr", 999.);
  desc.add<double>("l1LowerThrIgnoreIsolation", 0.0);
  desc.add<double>("regionEtaMargin", 0.14);
  desc.add<double>("regionPhiMargin", 0.4);
  //desc.add<std::vector<std::string> >("RecHitFlagToBeExcluded", std::vector<std::string>());
  //desc.add<std::vector<std::string> >("RecHitSeverityToBeExcluded", std::vector<std::string>());
  descriptions.add(("hltEgammaHLTRechitInRegionsProducer"), desc);  
}

void EgammaHLTRechitInRegionsProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;
  std::auto_ptr<const CaloSubdetectorTopology> topology;
    
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
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emIsolColl->begin(); emItr != emIsolColl->end() ;++emItr ) {
      if ((emItr->et() > l1LowerThr_) and (emItr->et() < l1UpperThr_)) {

	// Access the GCT hardware object corresponding to the L1Extra EM object.
	int etaIndex = emItr->gctEmCand()->etaIndex();
	int phiIndex = emItr->gctEmCand()->phiIndex();

	// Use the L1CaloGeometry to find the eta, phi bin boundaries.
	double etaLow  = l1CaloGeom->etaBinLowEdge(etaIndex);
	double etaHigh = l1CaloGeom->etaBinHighEdge(etaIndex);
	double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
	double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));
      }
    }
  }
  
  if(!doIsolated_ or (l1LowerThrIgnoreIsolation_ < 64)) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emNonIsolColl->begin(); emItr != emNonIsolColl->end() ;++emItr ) {
      
      if(doIsolated_ and (emItr->et() < l1LowerThrIgnoreIsolation_)) 
	continue;

      if ((emItr->et() > l1LowerThr_) and (emItr->et() < l1UpperThr_)) {
	
	// Access the GCT hardware object corresponding to the L1Extra EM object.
	int etaIndex = emItr->gctEmCand()->etaIndex();
	int phiIndex = emItr->gctEmCand()->phiIndex();
	
	// Use the L1CaloGeometry to find the eta, phi bin boundaries.
	double etaLow  = l1CaloGeom->etaBinLowEdge(etaIndex);
	double etaHigh = l1CaloGeom->etaBinHighEdge(etaIndex);
	double phiLow  = l1CaloGeom->emJetPhiBinLowEdge(phiIndex);
	double phiHigh = l1CaloGeom->emJetPhiBinHighEdge(phiIndex);

	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));
      }
    }
  }
  if (useUncalib_) {
    edm::Handle<EcalUncalibratedRecHitCollection> urhcH[3];
    for (unsigned int i=0; i<hitLabels.size(); i++) {
      std::auto_ptr<EcalUncalibratedRecHitCollection> uhits(new EcalUncalibratedRecHitCollection);
      
      evt.getByToken(uncalibHitTokens[i], urhcH[i]);  
      if (!(urhcH[i].isValid())) {
	edm::LogError("ProductNotFound")<< "could not get a handle on the EcalRecHitCollection! (" << hitLabels[i].encode() << ")" << std::endl;
	return;
      }
      const EcalUncalibratedRecHitCollection* uncalibRecHits = urhcH[i].product();
      
      if (uncalibRecHits->size() == 0)
	continue;
      
      if ((*uncalibRecHits)[0].id().subdetId() == EcalBarrel) {
	geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
	topology.reset(new EcalBarrelTopology(geoHandle));
      } else if ((*uncalibRecHits)[0].id().subdetId() == EcalEndcap) {
	geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
	topology.reset(new EcalEndcapTopology(geoHandle));
      } else if ((*uncalibRecHits)[0].id().subdetId() == EcalPreshower) {
	geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
	topology.reset(new EcalPreshowerTopology (geoHandle));
      } else throw(std::runtime_error("\n\nProducer encountered invalied ecalhitcollection type.\n\n"));
      
      if(regions.size() != 0) {
	EcalUncalibratedRecHitCollection::const_iterator it;
	
	for (it = uncalibRecHits->begin(); it != uncalibRecHits->end(); it++){
	  const CaloCellGeometry *this_cell = (*geometry_p).getGeometry(it->id());
	  GlobalPoint position = this_cell->getPosition();
	  
	  std::vector<EcalEtaPhiRegion>::const_iterator region;
	  for (region=regions.begin(); region!=regions.end(); region++) {
	    if (region->inRegion(position))
	      uhits->push_back(*it);
	  }
	}
      }

      evt.put(uhits, productLabels[i]);
    }
  } else {
    edm::Handle<EcalRecHitCollection> rhcH[3];
    
    for (unsigned int i=0; i<hitLabels.size(); i++) {
      std::auto_ptr<EcalRecHitCollection> hits(new EcalRecHitCollection);
    
      evt.getByToken(hitTokens[i], rhcH[i]);  
      if (!(rhcH[i].isValid())) {
	edm::LogError("ProductNotFound")<< "could not get a handle on the EcalRecHitCollection! (" << hitLabels[i].encode() << ")" << std::endl;
	return;
      }
      const EcalRecHitCollection* recHits = rhcH[i].product();
      
      if (recHits->size() == 0)
	continue;
      
      if ((*recHits)[0].id().subdetId() == EcalBarrel) {
	geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
	topology.reset(new EcalBarrelTopology(geoHandle));
      } else if ((*recHits)[0].id().subdetId() == EcalEndcap) {
	geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
	topology.reset(new EcalEndcapTopology(geoHandle));
      } else if ((*recHits)[0].id().subdetId() == EcalPreshower) {
	geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
	topology.reset(new EcalPreshowerTopology (geoHandle));
      } else throw(std::runtime_error("\n\nProducer encountered invalied ecalhitcollection type.\n\n"));
      
      if(regions.size() != 0) {
	EcalRecHitCollection::const_iterator it;	
	for (it = recHits->begin(); it != recHits->end(); it++){
	  const CaloCellGeometry *this_cell = (*geometry_p).getGeometry(it->id());
	  GlobalPoint position = this_cell->getPosition();
	  
	  std::vector<EcalEtaPhiRegion>::const_iterator region;
	  for (region=regions.begin(); region!=regions.end(); region++) {
	    if (region->inRegion(position))
	      hits->push_back(*it);
	  }
	}
      }

      evt.put(hits, productLabels[i]);
    }
  }
}


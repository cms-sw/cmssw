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


EgammaHLTRechitInRegionsProducer::EgammaHLTRechitInRegionsProducer(const edm::ParameterSet& ps) {

  hitproducer_   = ps.getParameter<edm::InputTag>("ecalhitproducer");

  l1TagIsolated_ = ps.getParameter< edm::InputTag > ("l1TagIsolated");
  l1TagNonIsolated_ = ps.getParameter< edm::InputTag > ("l1TagNonIsolated");
  doIsolated_   = ps.getParameter<bool>("doIsolated");
  
  l1LowerThr_ = ps.getParameter<double> ("l1LowerThr");
  l1UpperThr_ = ps.getParameter<double> ("l1UpperThr");
  l1LowerThrIgnoreIsolation_ = ps.getParameter<double> ("l1LowerThrIgnoreIsolation");

  regionEtaMargin_   = ps.getParameter<double>("regionEtaMargin");
  regionPhiMargin_   = ps.getParameter<double>("regionPhiMargin");

  const std::vector<std::string> flagnames = ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");
  const std::vector<int> flagsexcl = StringToEnumValue<EcalRecHit::Flags>(flagnames);
  
  const std::vector<std::string> severitynames = ps.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcluded");
  const std::vector<int> severitiesexcl = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  hitLabels = ps.getParameter<std::vector<edm::InputTag>>("ecalhitLabels");
  for (unsigned int i=0; i<hitLabels.size(); i++) 
    hitTokens.push_back(consumes<EcalRecHitCollection>(hitLabels[i]));
    
  produces<EcalRecHitCollection> ("EcalRegionalRecHitsEB");
  produces<EcalRecHitCollection> ("EcalRegionalRecHitsEE");
  produces<EcalRecHitCollection> ("EcalRegionalRecHitsES");

  nEvt_ = 0;
}


EgammaHLTRechitInRegionsProducer::~EgammaHLTRechitInRegionsProducer()
{}

void EgammaHLTRechitInRegionsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<edm::InputTag> inputTags;
  inputTags.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEB"));
  inputTags.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEE"));
  inputTags.push_back(edm::InputTag("hltESRegionalEgammaRecHit:EcalRecHitsES"));
  desc.add<std::vector<edm::InputTag>>("ecalhitLabels", inputTags);
  desc.add<edm::InputTag>("ecalhitproducer", edm::InputTag("ecalRecHit"));
  desc.add<edm::InputTag>("l1TagIsolated", edm::InputTag("l1extraParticles","Isolated"));
  desc.add<edm::InputTag>("l1TagNonIsolated", edm::InputTag("l1extraParticles","NonIsolated"));
  desc.add<bool>("doIsolated", true);
  desc.add<double>("l1LowerThr", 5.0);
  desc.add<double>("l1UpperThr", 999.);
  desc.add<double>("l1LowerThrIgnoreIsolation", 0.0);
  desc.add<double>("regionEtaMargin", 0.14);
  desc.add<double>("regionPhiMargin", 0.4);
  desc.add<std::vector<std::string> >("RecHitFlagToBeExcluded", std::vector<std::string>());
  desc.add<std::vector<std::string> >("RecHitSeverityToBeExcluded", std::vector<std::string>());
  descriptions.add(("hltEgammaHLTRechitInRegionsProducer"), desc);  
}

void EgammaHLTRechitInRegionsProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  std::auto_ptr<EcalRecHitCollection> hitsEB(new EcalRecHitCollection);
  std::auto_ptr<EcalRecHitCollection> hitsEE(new EcalRecHitCollection);
  std::auto_ptr<EcalRecHitCollection> hitsES(new EcalRecHitCollection);
  
  edm::Handle<EcalRecHitCollection> rhcH[3];

  for (unsigned int i=0; i<hitLabels.size(); i++) {
    evt.getByToken(hitTokens[i], rhcH[i]);  
    if (!(rhcH[i].isValid())) {
      edm::LogError("ProductNotFound")<< "could not get a handle on the EcalRecHitCollection! (" << hitLabels[i].encode() << ")" << std::endl;
      return;
    }
  }
  
  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;
  std::auto_ptr<const CaloSubdetectorTopology> topology;

  //edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  //es.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  //const EcalSeverityLevelAlgo* sevLevel = sevlv.product();
  
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
  
  for (unsigned int i=0; i<hitLabels.size(); i++) {
    const EcalRecHitCollection *recHits = rhcH[i].product();
    if(hitLabels[i].encode() == "hltEcalRegionalEgammaRecHit:EcalRecHitsEB") {
      geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
      topology.reset(new EcalBarrelTopology(geoHandle));
    } else if(hitLabels[i].encode() == "hltEcalRegionalEgammaRecHit:EcalRecHitsEE") {
      geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
      topology.reset(new EcalEndcapTopology(geoHandle));
    } else if(hitLabels[i].encode() == "hltESRegionalEgammaRecHit:EcalRecHitsES") {
      geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
      topology.reset(new EcalPreshowerTopology (geoHandle));
    } else throw(std::runtime_error("\n\nProducer encountered invalied ecalhitcollection type.\n\n"));

    if(regions.size() != 0) {
      EcalRecHitCollection::const_iterator it;

      for (it = recHits->begin(); it != recHits->end(); it++){
	//Make the vector of seeds that we're going to use.
	//One of the few places position is used, needed for ET calculation.    
	const CaloCellGeometry *this_cell = (*geometry_p).getGeometry(it->id());
	GlobalPoint position = this_cell->getPosition();

	std::vector<EcalEtaPhiRegion>::const_iterator region;
	for (region=regions.begin(); region!=regions.end(); region++) {
	  //std::cout << region->etaLow() << " " << region->etaHigh() <<  " " << region->phiLow() << " " << region->phiHigh() << std::endl;
	  //std::cout << i << " -  " << position.eta() << " " << position.phi() << std::endl;
	  if (region->inRegion(position)) {
	    if (i == 0)
	      hitsEB->push_back(*it);
	    if (i == 1)
	      hitsEE->push_back(*it);
	    if (i == 2)
	      hitsES->push_back(*it);
	  }
	}
      }
    }
  }

  //std::cout << hitsEB->size() << " " << hitsEE->size() << " " << hitsES->size() << std::endl;
  evt.put(hitsEB, "EcalRegionalRecHitsEB");
  evt.put(hitsEE, "EcalRegionalRecHitsEE");
  evt.put(hitsES, "EcalRegionalRecHitsES");

  nEvt_++;
}


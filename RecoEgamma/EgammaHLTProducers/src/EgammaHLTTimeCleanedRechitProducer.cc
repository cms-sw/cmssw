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

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTTimeCleanedRechitProducer.h"

EgammaHLTTimeCleanedRechitProducer::EgammaHLTTimeCleanedRechitProducer(const edm::ParameterSet& ps) {

  timeMax_ = ps.getParameter<double>("TimeMax");
  timeMin_ = ps.getParameter<double>("TimeMin");
  hitLabels     = ps.getParameter<std::vector<edm::InputTag>>("ecalhitLabels");
  productLabels = ps.getParameter<std::vector<std::string>>("productLabels");

   if (!(hitLabels.size() == 2)) throw cms::Exception("ConfigError","ecalhitLabels should contain 2 labels: one for RecHits in barrel, the other for RecHits in endcaps");
   if (!(productLabels.size() == 2)) throw cms::Exception("ConfigError","productLabels should contain 2 labels: one for RecHits in barrel, the other for RecHits in endcaps");

    
  for (unsigned int i=0; i<hitLabels.size(); i++) { 
    hitTokens.push_back(consumes<EcalRecHitCollection>(hitLabels[i]));
    produces<EcalRecHitCollection> (productLabels[i]);
  }
}


EgammaHLTTimeCleanedRechitProducer::~EgammaHLTTimeCleanedRechitProducer()
{}

void EgammaHLTTimeCleanedRechitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> productTags;
  productTags.push_back("EcalTimeCleanedRecHitsEB");
  productTags.push_back("EcalTimeCleanedRecHitsEE");
  desc.add<std::vector<std::string>>("productLabels", productTags);
  std::vector<edm::InputTag> inputTags;
  inputTags.push_back(edm::InputTag("hltEcalRecHitAll:EcalRecHitsEB"));
  inputTags.push_back(edm::InputTag("hltEcalRecHitAll:EcalRecHitsEE"));
  desc.add<std::vector<edm::InputTag>>("ecalhitLabels", inputTags);
  desc.add<double>("TimeMax", 10);
  desc.add<double>("TimeMin", -10);
  descriptions.add(("hltEgammaHLTTimeCleanedRechitProducer"), desc);
}

void EgammaHLTTimeCleanedRechitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  edm::Handle<EcalRecHitCollection> rhcH[2];
    

  for (unsigned int i=0; i<hitLabels.size(); i++) {
    std::auto_ptr<EcalRecHitCollection> hits(new EcalRecHitCollection);
    
    evt.getByToken(hitTokens[i], rhcH[i]);  
    if (!(rhcH[i].isValid())) {
      edm::LogError("ProductNotFound")<< "could not get a handle on the EcalRecHitCollection! (" << hitLabels[i].encode() << ")" << std::endl;
      return;
    }
    const EcalRecHitCollection* recHits = rhcH[i].product();
      
    EcalRecHitCollection::const_iterator it;	
    for (it = recHits->begin(); it != recHits->end(); it++){
      if (((*it).time() < timeMax_) && ((*it).time() > timeMin_))
	hits->push_back(*it);
    }
    
    evt.put(hits, productLabels[i]);
  }
}


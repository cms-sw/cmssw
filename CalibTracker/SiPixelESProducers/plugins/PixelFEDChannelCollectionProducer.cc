// -*- C++ -*-
//
// Package:    CalibTracker/PixelFEDChannelCollectionProducer
// Class:      PixelFEDChannelCollectionProducer
//
/**\class PixelFEDChannelCollectionProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 13 Dec 2018 08:48:22 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CalibTracker/Records/interface/SiPixelFEDChannelContainerESProducerRcd.h"

// Need to add #include statements for definitions of
// the data type and record type here

//
// class declaration
//

class PixelFEDChannelCollectionProducer : public edm::ESProducer {
public:
  PixelFEDChannelCollectionProducer(const edm::ParameterSet&);
  ~PixelFEDChannelCollectionProducer() override;

  typedef std::unordered_map<std::string, PixelFEDChannelCollection> PixelFEDChannelCollectionMap;
  using ReturnType = std::unique_ptr<PixelFEDChannelCollectionMap>;

  ReturnType produce(const SiPixelFEDChannelContainerESProducerRcd&);

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<SiPixelFEDChannelContainer, SiPixelStatusScenariosRcd> qualityToken_;
};

PixelFEDChannelCollectionProducer::PixelFEDChannelCollectionProducer(const edm::ParameterSet& iConfig)
    : qualityToken_(setWhatProduced(this).consumes()) {}

PixelFEDChannelCollectionProducer::~PixelFEDChannelCollectionProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
PixelFEDChannelCollectionProducer::ReturnType PixelFEDChannelCollectionProducer::produce(
    const SiPixelFEDChannelContainerESProducerRcd& iRecord) {
  const auto& qualityCollection = iRecord.get(qualityToken_);

  auto out = std::make_unique<PixelFEDChannelCollectionMap>();

  for (const auto& it : qualityCollection.getScenarioMap()) {
    const std::string& scenario = it.first;
    // getScenarioMap() is an unordered_map<string, ...>, so each scenario appears exactly once
    PixelFEDChannelCollection& disabled_channelcollection = (*out)[scenario];

    const auto& SiPixelBadFedChannels = it.second;
    for (const auto& entry : SiPixelBadFedChannels) {
      disabled_channelcollection.insert(entry.first, entry.second.data(), entry.second.size());
    }
  }

  return out;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(PixelFEDChannelCollectionProducer);

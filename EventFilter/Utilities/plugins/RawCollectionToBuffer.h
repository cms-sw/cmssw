#ifndef EventFilter_Utilities_RawCollectionToBuffer_h
#define EventFilter_Utilities_RawCollectionToBuffer_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
//#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
//#include "EventFilter/Utilities/interface/crc32c.h"
//#include "EventFilter/Utilities/plugins/EvFBuildingThrottle.h"
//#include "FWCore/Framework/interface/EventForOutput.h"
//#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
//#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class RawCollectionToBuffer : public edm::one::EDProducer<> {
public:
  explicit RawCollectionToBuffer(edm::ParameterSet const& ps);
  ~RawCollectionToBuffer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& e, edm::EventSetup const&) override;
  const edm::EDGetTokenT<FEDRawDataCollection> token_;
  //std::vector<unsigned int> sourceIdList_;
};

inline RawCollectionToBuffer::RawCollectionToBuffer(edm::ParameterSet const& ps)
    : token_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("source"))) {
  //sourceIdList_(ps.getUntrackedParameter<std::vector<unsigned int>>("sourceIdList", std::vector<unsigned int>())) {
  produces<RawDataBuffer>();
}

inline void RawCollectionToBuffer::produce(edm::Event& e, edm::EventSetup const&) {
  edm::Handle<FEDRawDataCollection> collection;
  e.getByToken(token_, collection);
  uint32_t totalSize = 0;
  int nFeds = FEDNumbering::lastFEDId() + 1;
  for (int idx = 0; idx < nFeds; ++idx) {
    totalSize += collection->FEDData(idx).size();
  }
  std::unique_ptr<RawDataBuffer> rawBuffer = std::make_unique<RawDataBuffer>(totalSize);
  rawBuffer->setPhase1Range();
  for (int idx = 0; idx < nFeds; ++idx) {
    auto size = collection->FEDData(idx).size();
    if (size)
      rawBuffer->addSource(idx, collection->FEDData(idx).data(), size);
  }
  e.put(std::move(rawBuffer));
}

inline void RawCollectionToBuffer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("rawDataCollector"));
  //desc.addUntracked<std::vector<unsigned int>>("sourceIdList", std::vector<unsigned int>());
  descriptions.addWithDefaultLabel(desc);
}

#endif  // IOPool_Streamer_interface_RawCollectionToBuffer_h

#ifndef EventFilter_Utilities_RawBufferToCollection_h
#define EventFilter_Utilities_RawBufferToCollection_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class RawBufferToCollection : public edm::one::EDProducer<> {
public:
  explicit RawBufferToCollection(edm::ParameterSet const& ps);
  ~RawBufferToCollection() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& e, edm::EventSetup const&) override;
  const edm::EDGetTokenT<RawDataBuffer> token_;
};

inline RawBufferToCollection::RawBufferToCollection(edm::ParameterSet const& ps)
    : token_(consumes<RawDataBuffer>(ps.getParameter<edm::InputTag>("source"))) {
  produces<FEDRawDataCollection>();
}

inline void RawBufferToCollection::produce(edm::Event& e, edm::EventSetup const&) {
  edm::Handle<RawDataBuffer> fedBuffer;
  e.getByToken(token_, fedBuffer);

  std::unique_ptr<FEDRawDataCollection> collection = std::make_unique<FEDRawDataCollection>();

  for (auto it = fedBuffer->map().begin(); it != fedBuffer->map().end(); it++) {
    auto singleFED = fedBuffer->fragmentData(it);
    if (it->first > FEDNumbering::lastFEDId())
      throw cms::Exception("RawBufferToCollection")
          << "FED " << it->first << " exceeds FEDRawDataCollection maximum FED " << FEDNumbering::lastFEDId();
    FEDRawData& fedData = collection->FEDData(it->first);
    fedData.resize(singleFED.size());
    memcpy(fedData.data(), &singleFED.data()[0], singleFED.size());
  }
  e.put(std::move(collection));
}

inline void RawBufferToCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("rawDataCollector"));
  descriptions.addWithDefaultLabel(desc);
}

#endif  // IOPool_Streamer_interface_RawBufferToCollection_h

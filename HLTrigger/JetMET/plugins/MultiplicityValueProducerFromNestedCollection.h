#ifndef HLTrigger_JetMET_plugins_MultiplicityValueProducerFromNestedCollection_h
#define HLTrigger_JetMET_plugins_MultiplicityValueProducerFromNestedCollection_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <class INP_TYPE, class OUT_TYPE>
class MultiplicityValueProducerFromNestedCollection : public edm::global::EDProducer<> {
public:
  explicit MultiplicityValueProducerFromNestedCollection(edm::ParameterSet const&);
  ~MultiplicityValueProducerFromNestedCollection() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  edm::EDGetTokenT<INP_TYPE> const src_token_;
  OUT_TYPE const defaultValue_;
};

template <class INP_TYPE, class OUT_TYPE>
MultiplicityValueProducerFromNestedCollection<INP_TYPE, OUT_TYPE>::MultiplicityValueProducerFromNestedCollection(
    edm::ParameterSet const& iConfig)
    : src_token_(consumes<INP_TYPE>(iConfig.getParameter<edm::InputTag>("src"))),
      defaultValue_(iConfig.getParameter<OUT_TYPE>("defaultValue")) {
  produces<OUT_TYPE>();
}

template <class INP_TYPE, class OUT_TYPE>
void MultiplicityValueProducerFromNestedCollection<INP_TYPE, OUT_TYPE>::produce(edm::StreamID,
                                                                                edm::Event& iEvent,
                                                                                edm::EventSetup const& iSetup) const {
  auto const& objHandle(iEvent.getHandle(src_token_));

  if (objHandle.isValid()) {
    LogDebug("Input") << "size of input collection: " << objHandle->size();

    OUT_TYPE objMult(0);
    for (auto const& obj : *objHandle) {
      objMult += obj.size();
    }

    LogDebug("Output") << "size of output objects: " << objMult;

    iEvent.put(std::make_unique<OUT_TYPE>(objMult));
  } else {
    iEvent.put(std::make_unique<OUT_TYPE>(defaultValue_));
  }
}

template <class INP_TYPE, class OUT_TYPE>
void MultiplicityValueProducerFromNestedCollection<INP_TYPE, OUT_TYPE>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input collection");
  desc.add<OUT_TYPE>("defaultValue")->setComment("default output value (used when input collection is unavailable)");
  descriptions.addWithDefaultLabel(desc);
}

#endif  // HLTrigger_JetMET_plugins_MultiplicityValueProducerFromNestedCollection_h

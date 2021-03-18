#ifndef HLTrigger_JetMET_plugins_HLTMultiplicityValueProducer_h
#define HLTrigger_JetMET_plugins_HLTMultiplicityValueProducer_h

#include <string>
#include <memory>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <class INP_TYPE, class OUT_TYPE>
class HLTMultiplicityValueProducer : public edm::global::EDProducer<> {
public:
  explicit HLTMultiplicityValueProducer(edm::ParameterSet const&);
  ~HLTMultiplicityValueProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  edm::EDGetTokenT<edm::View<INP_TYPE>> const src_token_;
  StringCutObjectSelector<INP_TYPE, true> const strObjSelector_;
  OUT_TYPE const defaultValue_;
};

template <class INP_TYPE, class OUT_TYPE>
HLTMultiplicityValueProducer<INP_TYPE, OUT_TYPE>::HLTMultiplicityValueProducer(edm::ParameterSet const& iConfig)
    : src_token_(consumes<edm::View<INP_TYPE>>(iConfig.getParameter<edm::InputTag>("src"))),
      strObjSelector_(StringCutObjectSelector<INP_TYPE, true>(iConfig.getParameter<std::string>("cut"))),
      defaultValue_(iConfig.getParameter<OUT_TYPE>("defaultValue")) {
  produces<OUT_TYPE>();
}

template <class INP_TYPE, class OUT_TYPE>
void HLTMultiplicityValueProducer<INP_TYPE, OUT_TYPE>::produce(edm::StreamID,
                                                               edm::Event& iEvent,
                                                               edm::EventSetup const& iSetup) const {
  auto const& objHandle(iEvent.getHandle(src_token_));

  if (objHandle.isValid()) {
    LogDebug("Input") << "size of input collection: " << objHandle->size();

    OUT_TYPE objMult(0);
    for (auto const& obj : *objHandle) {
      if (strObjSelector_(obj)) {
        ++objMult;
      }
    }

    LogDebug("Output") << "size of selected input objects: " << objMult;

    iEvent.put(std::make_unique<OUT_TYPE>(objMult));
  } else {
    iEvent.put(std::make_unique<OUT_TYPE>(defaultValue_));
  }
}

template <class INP_TYPE, class OUT_TYPE>
void HLTMultiplicityValueProducer<INP_TYPE, OUT_TYPE>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag(""))->setComment("input collection");
  desc.add<std::string>("cut", "")->setComment("string for StringCutObjectSelector");
  desc.add<OUT_TYPE>("defaultValue", 0)->setComment("default output value (used when input collection is unavailable)");
  descriptions.addWithDefaultLabel(desc);
}

#endif  // HLTrigger_JetMET_plugins_HLTMultiplicityValueProducer_h

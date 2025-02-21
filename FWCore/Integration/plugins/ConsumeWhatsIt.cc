// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      ConsumeWhatsIt
//
/**\class edmtest::ConsumeWhatsIt

 Description:
     Consumes and produces EventSetup products.
     This is used to test the printout from the Tracer
     module related to EventSetup module dependencies.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  11 December 2024

#include <memory>
#include <optional>

#include "Doodad.h"
#include "GadgetRcd.h"
#include "WhatsIt.h"

#include "FWCore/Framework/interface/es_Label.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

namespace edmtest {

  class ConsumeWhatsIt : public edm::ESProducer {
  public:
    ConsumeWhatsIt(edm::ParameterSet const& pset);

    using ReturnType = std::unique_ptr<WhatsIt>;
    using ReturnTypeA = std::unique_ptr<const WhatsIt>;
    using ReturnTypeB = std::shared_ptr<WhatsIt>;
    using ReturnTypeC = std::shared_ptr<const WhatsIt>;
    using ReturnTypeD = std::optional<WhatsIt>;

    ReturnType produce(const GadgetRcd&);
    ReturnTypeA produceA(const GadgetRcd&);
    ReturnTypeB produceB(const GadgetRcd&);
    ReturnTypeC produceC(const GadgetRcd&);
    ReturnTypeD produceD(const GadgetRcd&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ESGetToken<Doodad, GadgetRcd> token_;
    edm::ESGetToken<Doodad, GadgetRcd> tokenA_;
    edm::ESGetToken<Doodad, GadgetRcd> tokenB_;
    edm::ESGetToken<Doodad, GadgetRcd> tokenC_;
    edm::ESGetToken<Doodad, GadgetRcd> tokenD_;

    edm::ESGetToken<WhatsIt, GadgetRcd> token_in_produce_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenA_in_produce_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenB_in_produce_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenC_in_produce_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenD_in_produce_;

    edm::ESGetToken<WhatsIt, GadgetRcd> token_in_produceA_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenA_in_produceA_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenB_in_produceA_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenC_in_produceA_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenD_in_produceA_;

    edm::ESGetToken<WhatsIt, GadgetRcd> token_in_produceB_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenA_in_produceB_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenB_in_produceB_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenC_in_produceB_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenD_in_produceB_;

    edm::ESGetToken<WhatsIt, GadgetRcd> token_in_produceC_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenA_in_produceC_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenB_in_produceC_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenC_in_produceC_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenD_in_produceC_;

    edm::ESGetToken<WhatsIt, GadgetRcd> token_in_produceD_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenA_in_produceD_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenB_in_produceD_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenC_in_produceD_;
    edm::ESGetToken<WhatsIt, GadgetRcd> tokenD_in_produceD_;
  };

  ConsumeWhatsIt::ConsumeWhatsIt(edm::ParameterSet const& pset) {
    auto collector = setWhatProduced(this, "R");
    auto collectorA = setWhatProduced(this, &ConsumeWhatsIt::produceA, edm::es::Label("AR"));
    auto collectorB = setWhatProduced(this, &ConsumeWhatsIt::produceB, edm::es::Label("BR"));
    auto collectorC = setWhatProduced(this, &ConsumeWhatsIt::produceC, edm::es::Label("CR"));
    auto collectorD = setWhatProduced(this, &ConsumeWhatsIt::produceD, edm::es::Label("DR"));

    token_ = collector.consumes(edm::ESInputTag{"", ""});
    tokenA_ = collectorA.consumes(edm::ESInputTag{"", ""});
    tokenB_ = collectorB.consumes(edm::ESInputTag{"", ""});
    tokenC_ = collectorC.consumes(edm::ESInputTag{"", ""});
    tokenD_ = collectorD.consumes(edm::ESInputTag{"", ""});

    token_in_produce_ = collector.consumes(pset.getParameter<edm::ESInputTag>("esInputTag_in_produce"));
    tokenA_in_produce_ = collector.consumes(pset.getParameter<edm::ESInputTag>("esInputTagA_in_produce"));
    tokenB_in_produce_ = collector.consumes(pset.getParameter<edm::ESInputTag>("esInputTagB_in_produce"));
    tokenC_in_produce_ = collector.consumes(pset.getParameter<edm::ESInputTag>("esInputTagC_in_produce"));
    tokenD_in_produce_ = collector.consumes(pset.getParameter<edm::ESInputTag>("esInputTagD_in_produce"));

    token_in_produceA_ = collectorA.consumes(pset.getParameter<edm::ESInputTag>("esInputTag_in_produceA"));
    tokenA_in_produceA_ = collectorA.consumes(pset.getParameter<edm::ESInputTag>("esInputTagA_in_produceA"));
    tokenB_in_produceA_ = collectorA.consumes(pset.getParameter<edm::ESInputTag>("esInputTagB_in_produceA"));
    tokenC_in_produceA_ = collectorA.consumes(pset.getParameter<edm::ESInputTag>("esInputTagC_in_produceA"));
    tokenD_in_produceA_ = collectorA.consumes(pset.getParameter<edm::ESInputTag>("esInputTagD_in_produceA"));

    token_in_produceB_ = collectorB.consumes(pset.getParameter<edm::ESInputTag>("esInputTag_in_produceB"));
    tokenA_in_produceB_ = collectorB.consumes(pset.getParameter<edm::ESInputTag>("esInputTagA_in_produceB"));
    tokenB_in_produceB_ = collectorB.consumes(pset.getParameter<edm::ESInputTag>("esInputTagB_in_produceB"));
    tokenC_in_produceB_ = collectorB.consumes(pset.getParameter<edm::ESInputTag>("esInputTagC_in_produceB"));
    tokenD_in_produceB_ = collectorB.consumes(pset.getParameter<edm::ESInputTag>("esInputTagD_in_produceB"));

    token_in_produceC_ = collectorC.consumes(pset.getParameter<edm::ESInputTag>("esInputTag_in_produceC"));
    tokenA_in_produceC_ = collectorC.consumes(pset.getParameter<edm::ESInputTag>("esInputTagA_in_produceC"));
    tokenB_in_produceC_ = collectorC.consumes(pset.getParameter<edm::ESInputTag>("esInputTagB_in_produceC"));
    tokenC_in_produceC_ = collectorC.consumes(pset.getParameter<edm::ESInputTag>("esInputTagC_in_produceC"));
    tokenD_in_produceC_ = collectorC.consumes(pset.getParameter<edm::ESInputTag>("esInputTagD_in_produceC"));

    token_in_produceD_ = collectorD.consumes(pset.getParameter<edm::ESInputTag>("esInputTag_in_produceD"));
    tokenA_in_produceD_ = collectorD.consumes(pset.getParameter<edm::ESInputTag>("esInputTagA_in_produceD"));
    tokenB_in_produceD_ = collectorD.consumes(pset.getParameter<edm::ESInputTag>("esInputTagB_in_produceD"));
    tokenC_in_produceD_ = collectorD.consumes(pset.getParameter<edm::ESInputTag>("esInputTagC_in_produceD"));
    tokenD_in_produceD_ = collectorD.consumes(pset.getParameter<edm::ESInputTag>("esInputTagD_in_produceD"));
  }

  ConsumeWhatsIt::ReturnType ConsumeWhatsIt::produce(const GadgetRcd& iRecord) {
    // For purposes of the tests this is currently intended for, we don't
    // need to actually get the data. We only need to call consumes.
    // At least initially this is only intended for testing the dumpPathsAndConsumes
    // option of the Tracer. Possibly it may be useful to extended this in the future...
    auto pWhatsIt = std::make_unique<WhatsIt>();
    return pWhatsIt;
  }

  ConsumeWhatsIt::ReturnTypeA ConsumeWhatsIt::produceA(const GadgetRcd& iRecord) {
    auto pWhatsIt = std::make_unique<WhatsIt>();
    return pWhatsIt;
  }

  ConsumeWhatsIt::ReturnTypeB ConsumeWhatsIt::produceB(const GadgetRcd& iRecord) {
    auto pWhatsIt = std::make_shared<WhatsIt>();
    return pWhatsIt;
  }

  ConsumeWhatsIt::ReturnTypeC ConsumeWhatsIt::produceC(const GadgetRcd& iRecord) {
    auto pWhatsIt = std::make_shared<WhatsIt>();
    return pWhatsIt;
  }

  ConsumeWhatsIt::ReturnTypeD ConsumeWhatsIt::produceD(const GadgetRcd& iRecord) {
    auto pWhatsIt = std::make_optional<WhatsIt>();
    return pWhatsIt;
  }

  void ConsumeWhatsIt::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::ESInputTag>("esInputTag_in_produce");
    desc.add<edm::ESInputTag>("esInputTagA_in_produce");
    desc.add<edm::ESInputTag>("esInputTagB_in_produce");
    desc.add<edm::ESInputTag>("esInputTagC_in_produce");
    desc.add<edm::ESInputTag>("esInputTagD_in_produce");

    desc.add<edm::ESInputTag>("esInputTag_in_produceA");
    desc.add<edm::ESInputTag>("esInputTagA_in_produceA");
    desc.add<edm::ESInputTag>("esInputTagB_in_produceA");
    desc.add<edm::ESInputTag>("esInputTagC_in_produceA");
    desc.add<edm::ESInputTag>("esInputTagD_in_produceA");

    desc.add<edm::ESInputTag>("esInputTag_in_produceB");
    desc.add<edm::ESInputTag>("esInputTagA_in_produceB");
    desc.add<edm::ESInputTag>("esInputTagB_in_produceB");
    desc.add<edm::ESInputTag>("esInputTagC_in_produceB");
    desc.add<edm::ESInputTag>("esInputTagD_in_produceB");

    desc.add<edm::ESInputTag>("esInputTag_in_produceC");
    desc.add<edm::ESInputTag>("esInputTagA_in_produceC");
    desc.add<edm::ESInputTag>("esInputTagB_in_produceC");
    desc.add<edm::ESInputTag>("esInputTagC_in_produceC");
    desc.add<edm::ESInputTag>("esInputTagD_in_produceC");

    desc.add<edm::ESInputTag>("esInputTag_in_produceD");
    desc.add<edm::ESInputTag>("esInputTagA_in_produceD");
    desc.add<edm::ESInputTag>("esInputTagB_in_produceD");
    desc.add<edm::ESInputTag>("esInputTagC_in_produceD");
    desc.add<edm::ESInputTag>("esInputTagD_in_produceD");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_MODULE(ConsumeWhatsIt);

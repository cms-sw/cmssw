// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      MayConsumeWhatsIt
//
/**\class edmtest::MayConsumeWhatsIt

 Description:
     Consumes and produces EventSetup products.
     This test module is similar to ConsumesWhatsIt,
     but is focused on the "may consumes" variants
     of the the consumes interface.
     This is used to test the printout from the Tracer
     module related to EventSetup module dependencies.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  27 December 2024

#include <memory>

#include "Doodad.h"
#include "GadgetRcd.h"
#include "WhatsIt.h"

#include "FWCore/Framework/interface/es_Label.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTagGetter.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/IOVTestInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESProductTag.h"

namespace edmtest {

  class MayConsumeWhatsIt : public edm::ESProducer {
  public:
    MayConsumeWhatsIt(edm::ParameterSet const&);

    using ReturnType = std::unique_ptr<IOVTestInfo>;

    ReturnType produce(const GadgetRcd&);
    ReturnType produceMayConsumeTestDataA(const GadgetRcd&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ESGetToken<WhatsIt, GadgetRcd> token_for_produce_;
    edm::ESGetToken<ESTestDataA, GadgetRcd> token_for_produceMayConsumeTestDataA_;
  };

  MayConsumeWhatsIt::MayConsumeWhatsIt(edm::ParameterSet const& pset) {
    auto collector = setWhatProduced(this, edm::es::Label("DependsOnMayConsume"));
    collector.setMayConsume(
        token_for_produce_,
        [](edm::ESTagGetter get, edm::ESTransientHandle<Doodad> handle) { return get("", "DR"); },
        edm::ESProductTag<Doodad, GadgetRcd>("", ""));

    auto collector2 = setWhatProduced(
        this, &edmtest::MayConsumeWhatsIt::produceMayConsumeTestDataA, edm::es::Label("DependsOnMayConsume2"));
    collector2.setMayConsume(
        token_for_produceMayConsumeTestDataA_,
        [](edm::ESTagGetter get, edm::ESTransientHandle<Doodad> handle) { return get.nothing(); },
        edm::ESProductTag<Doodad, GadgetRcd>("", ""));
  }

  MayConsumeWhatsIt::ReturnType MayConsumeWhatsIt::produce(const GadgetRcd& iRecord) {
    auto handle = iRecord.getHandle(token_for_produce_);
    if (!handle.isValid()) {
      throw cms::Exception("TestFailure") << "MayConsumeWhatsIt::produceMayConsumeTestDataA, expected valid handle";
    }

    auto product = std::make_unique<IOVTestInfo>();
    return product;
  }

  MayConsumeWhatsIt::ReturnType MayConsumeWhatsIt::produceMayConsumeTestDataA(const GadgetRcd& iRecord) {
    // In the test, there will not be an ESProducer or ESSource that
    // produces ESTestDataA. The purpose is test the output from the
    // Tracer in that case.

    auto handle = iRecord.getHandle(token_for_produceMayConsumeTestDataA_);
    if (handle.isValid()) {
      throw cms::Exception("TestFailure") << "MayConsumeWhatsIt::produceMayConsumeTestDataA, expected invalid handle";
    }

    auto product = std::make_unique<IOVTestInfo>();
    return product;
  }

  void MayConsumeWhatsIt::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addDefault(desc);
  }

}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_EVENTSETUP_MODULE(MayConsumeWhatsIt);

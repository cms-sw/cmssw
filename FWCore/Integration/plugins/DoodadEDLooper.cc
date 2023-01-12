#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Doodad.h"
#include "GadgetRcd.h"

namespace edmtest {
  class DoodadEDLooper : public edm::ESProducerLooper {
  public:
    DoodadEDLooper(edm::ParameterSet const& iPSet) : token_(esConsumes<edm::Transition::BeginRun>()) {}
    ~DoodadEDLooper() override = default;

    void startingNewLoop(unsigned int) override {}

    void beginOfJob(edm::EventSetup const& iSetup) override {
      auto& doodad = iSetup.getData(token_);
      if (doodad.a != 1) {
        throw cms::Exception("TestFailure") << "beginOfJob: got " << doodad.a << " while expected 1";
      }
    }

    void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override {
      auto& doodad = iSetup.getData(token_);
      if (doodad.a != 1) {
        throw cms::Exception("TestFailure") << "beginRun: got " << doodad.a << " while expected 1";
      }
    }

    Status duringLoop(edm::Event const& iEvent, edm::EventSetup const& iSetup) override { return kContinue; }

    Status endOfLoop(edm::EventSetup const&, unsigned int iCount) override { return iCount == 2 ? kStop : kContinue; }

  private:
    edm::ESGetToken<Doodad, GadgetRcd> token_;
  };
}  // namespace edmtest

using namespace edmtest;
DEFINE_FWK_LOOPER(DoodadEDLooper);

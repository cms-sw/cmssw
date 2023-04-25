// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     ThingAnalyzer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Fri, 21 Apr 2017 13:34:58 GMT
//

// system include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// user include files

namespace edmtest {
  struct Empty {};
  class ThingAnalyzer : public edm::global::EDAnalyzer<edm::RunCache<Empty>, edm::LuminosityBlockCache<Empty>> {
  public:
    ThingAnalyzer(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final;
    std::shared_ptr<Empty> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final;
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const final;
    std::shared_ptr<Empty> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final;
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void shouldBeInvalid(edm::Handle<ThingCollection> const&) const;

    edm::EDGetTokenT<ThingCollection> beginRun_;
    edm::EDGetTokenT<ThingCollection> beginLumi_;
    edm::EDGetTokenT<ThingCollection> event_;
    edm::EDGetTokenT<ThingCollection> endLumi_;
    edm::EDGetTokenT<ThingCollection> endRun_;
  };

  ThingAnalyzer::ThingAnalyzer(edm::ParameterSet const& iPSet)
      : beginRun_(consumes<edm::InRun>(iPSet.getUntrackedParameter<edm::InputTag>("beginRun"))),
        beginLumi_(consumes<edm::InLumi>(iPSet.getUntrackedParameter<edm::InputTag>("beginLumi"))),
        event_(consumes(iPSet.getUntrackedParameter<edm::InputTag>("event"))),
        endLumi_(consumes<edm::InLumi>(iPSet.getUntrackedParameter<edm::InputTag>("endLumi"))),
        endRun_(consumes<edm::InRun>(iPSet.getUntrackedParameter<edm::InputTag>("endRun"))) {}

  void ThingAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked("beginRun", edm::InputTag{"thing", "beginRun"})->setComment("Collection to get from Run");
    desc.addUntracked("beginLumi", edm::InputTag{"thing", "beginLumi"})->setComment("Collection to get from Lumi");
    desc.addUntracked("event", edm::InputTag{"thing", ""})->setComment("Collection to get from event");
    desc.addUntracked("endLumi", edm::InputTag{"thing", "endLumi"})
        ->setComment("Collection to get from Lumi but only available at end");
    desc.addUntracked("endRun", edm::InputTag{"thing", "endRun"})
        ->setComment("Collection to get from Run but only available at end");
    descriptions.add("thingAnalyzer", desc);
  }

  void ThingAnalyzer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& lumi = iEvent.getLuminosityBlock();

    auto const& run = lumi.getRun();

    (void)run.get(beginRun_);

    shouldBeInvalid(run.getHandle(endRun_));

    (void)lumi.get(beginLumi_);

    shouldBeInvalid(lumi.getHandle(endLumi_));

    (void)iEvent.get(event_);
  }

  std::shared_ptr<Empty> ThingAnalyzer::globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const {
    (void)iRun.get(beginRun_);

    shouldBeInvalid(iRun.getHandle(endRun_));

    return std::shared_ptr<Empty>();
  }

  void ThingAnalyzer::globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const {
    (void)iRun.get(beginRun_);

    (void)iRun.get(endRun_);
  }

  std::shared_ptr<Empty> ThingAnalyzer::globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                   edm::EventSetup const&) const {
    auto const& run = iLumi.getRun();

    (void)run.get(beginRun_);

    shouldBeInvalid(run.getHandle(endRun_));

    (void)iLumi.get(beginLumi_);

    shouldBeInvalid(iLumi.getHandle(endLumi_));

    return std::shared_ptr<Empty>();
  }

  void ThingAnalyzer::globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) const {
    auto const& run = iLumi.getRun();

    (void)run.get(beginRun_);

    shouldBeInvalid(run.getHandle(endRun_));

    (void)iLumi.get(beginLumi_);

    (void)iLumi.get(endLumi_);
  }

  void ThingAnalyzer::shouldBeInvalid(edm::Handle<ThingCollection> const& iHandle) const {
    if (iHandle.isValid()) {
      throw cms::Exception("ShouldNotBeValid") << "handle was valid when it should not have been";
    }
  }

}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::ThingAnalyzer);

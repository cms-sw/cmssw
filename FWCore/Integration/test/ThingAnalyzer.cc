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

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override final;
    std::shared_ptr<Empty> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override final;
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override final;
    std::shared_ptr<Empty> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                      edm::EventSetup const&) const override final;
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override final;

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
      : beginRun_(consumes<ThingCollection, edm::InRun>(iPSet.getUntrackedParameter<edm::InputTag>("beginRun"))),
        beginLumi_(consumes<ThingCollection, edm::InLumi>(iPSet.getUntrackedParameter<edm::InputTag>("beginLumi"))),
        event_(consumes<ThingCollection>(iPSet.getUntrackedParameter<edm::InputTag>("event"))),
        endLumi_(consumes<ThingCollection, edm::InLumi>(iPSet.getUntrackedParameter<edm::InputTag>("endLumi"))),
        endRun_(consumes<ThingCollection, edm::InRun>(iPSet.getUntrackedParameter<edm::InputTag>("endRun"))) {}

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

    edm::Handle<ThingCollection> h;
    run.getByToken(beginRun_, h);
    *h;

    run.getByToken(endRun_, h);
    shouldBeInvalid(h);

    lumi.getByToken(beginLumi_, h);
    *h;

    lumi.getByToken(endLumi_, h);
    shouldBeInvalid(h);

    iEvent.getByToken(event_, h);
    *h;
  }

  std::shared_ptr<Empty> ThingAnalyzer::globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const {
    edm::Handle<ThingCollection> h;
    iRun.getByToken(beginRun_, h);
    *h;

    iRun.getByToken(endRun_, h);
    shouldBeInvalid(h);

    return std::shared_ptr<Empty>();
  }

  void ThingAnalyzer::globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const {
    edm::Handle<ThingCollection> h;
    iRun.getByToken(beginRun_, h);
    *h;

    iRun.getByToken(endRun_, h);
    *h;
  }

  std::shared_ptr<Empty> ThingAnalyzer::globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                   edm::EventSetup const&) const {
    auto const& run = iLumi.getRun();

    edm::Handle<ThingCollection> h;
    run.getByToken(beginRun_, h);
    *h;

    run.getByToken(endRun_, h);
    shouldBeInvalid(h);

    iLumi.getByToken(beginLumi_, h);
    *h;

    iLumi.getByToken(endLumi_, h);
    shouldBeInvalid(h);

    return std::shared_ptr<Empty>();
  }

  void ThingAnalyzer::globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) const {
    auto const& run = iLumi.getRun();

    edm::Handle<ThingCollection> h;
    run.getByToken(beginRun_, h);
    *h;

    run.getByToken(endRun_, h);
    shouldBeInvalid(h);

    iLumi.getByToken(beginLumi_, h);
    *h;

    iLumi.getByToken(endLumi_, h);
    *h;
  }

  void ThingAnalyzer::shouldBeInvalid(edm::Handle<ThingCollection> const& iHandle) const {
    if (iHandle.isValid()) {
      throw cms::Exception("ShouldNotBeValid") << "handle was valid when it should not have been";
    }
  }
}

DEFINE_FWK_MODULE(edmtest::ThingAnalyzer);

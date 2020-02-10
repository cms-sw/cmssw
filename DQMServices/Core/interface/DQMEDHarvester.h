#ifndef CORE_DQMED_HARVESTER_H
#define CORE_DQMED_HARVESTER_H

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/InputTagMatch.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/Histograms/interface/DQMToken.h"

namespace edm {
  class VInputTagMatch {
  public:
    VInputTagMatch(std::vector<edm::InputTag> const &inputTags) {
      for (auto &tag : inputTags) {
        matchers_.emplace_back(InputTagMatch(tag));
      }
    }

    bool operator()(edm::BranchDescription const &branchDescription) {
      for (auto &m : matchers_) {
        if (m(branchDescription)) {
          return true;
        }
      }
      return false;
    }

  private:
    std::vector<InputTagMatch> matchers_;
  };
}  // namespace edm

class DQMEDHarvester
    : public edm::one::EDProducer<edm::EndLuminosityBlockProducer,
                                  edm::EndRunProducer,
                                  edm::one::WatchLuminosityBlocks,
                                  edm::one::WatchRuns,
                                  // for uncontrolled DQMStore access, and that EDM does not even attempt to
                                  // run things in parallel (which would then be blocked by the booking lock).
                                  edm::one::SharedResources,
                                  edm::Accumulator> {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

protected:
  DQMStore *dqmstore_;
  edm::GetterOfProducts<DQMToken> runmegetter_;
  edm::GetterOfProducts<DQMToken> lumimegetter_;
  edm::EDPutTokenT<DQMToken> lumiToken_;
  edm::EDPutTokenT<DQMToken> runToken_;

public:
  DQMEDHarvester(edm::ParameterSet const &iConfig) {
    usesResource("DQMStore");
    dqmstore_ = edm::Service<DQMStore>().operator->();

    auto inputgeneration = iConfig.getUntrackedParameter<std::string>("inputGeneration", "DQMGenerationReco");
    auto outputgeneration = iConfig.getUntrackedParameter<std::string>("outputGeneration", "DQMGenerationHarvesting");

    // TODO: Run/Lumi suffix should not be needed, complain to CMSSW core in case.
    lumiToken_ = produces<DQMToken, edm::Transition::EndLuminosityBlock>(outputgeneration + "Lumi");
    runToken_ = produces<DQMToken, edm::Transition::EndRun>(outputgeneration + "Run");

    // Use explicitly specified inputs, but if there are none...
    auto inputtags =
        iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("inputMEs", std::vector<edm::InputTag>());
    if (inputtags.empty()) {
      // ... use all RECO MEs.
      inputtags.push_back(edm::InputTag("", inputgeneration + "Run"));
      inputtags.push_back(edm::InputTag("", inputgeneration + "Lumi"));
    }
    runmegetter_ = edm::GetterOfProducts<DQMToken>(edm::VInputTagMatch(inputtags), this, edm::InRun);
    lumimegetter_ = edm::GetterOfProducts<DQMToken>(edm::VInputTagMatch(inputtags), this, edm::InLumi);
    callWhenNewProductsRegistered([this](edm::BranchDescription const &bd) {
      runmegetter_(bd);
      lumimegetter_(bd);
    });
  };

  DQMEDHarvester() : DQMEDHarvester(edm::ParameterSet()){};

  void beginJob() override{};

  void beginRun(edm::Run const &run, edm::EventSetup const &) override {
    // According to edm experts, it is never save to look at run products
    // in beginRun, since they might be merged as new input files how up.
  }

  void beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &) final {
    // According to edm experts, it is never save to look at run products
    // in beginRun, since they might be merged as new input files how up.
  }

  void accumulate(edm::Event const &ev, edm::EventSetup const &es) final {
    dqmstore_->meBookerGetter([this, &ev, &es](DQMStore::IBooker &b, DQMStore::IGetter &g) {
      b.setScope(MonitorElementData::Scope::JOB);
      this->dqmAnalyze(b, g, ev, es);
    });
  }

  void endLuminosityBlockProduce(edm::LuminosityBlock &lumi, edm::EventSetup const &es) final {
    // No need to actually get products for now
    //auto refs = std::vector<edm::Handle<DQMToken>>();
    //lumimegetter_.fillHandles(lumi, refs);

    dqmstore_->meBookerGetter([this, &lumi, &es](DQMStore::IBooker &b, DQMStore::IGetter &g) {
      b.setScope(MonitorElementData::Scope::JOB);
      this->dqmEndLuminosityBlock(b, g, lumi, es);
    });

    lumi.put(lumiToken_, std::make_unique<DQMToken>());
  }

  void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) final{};

  void endRunProduce(edm::Run &run, edm::EventSetup const &es) final {
    dqmstore_->meBookerGetter([this, &run, &es](DQMStore::IBooker &b, DQMStore::IGetter &g) {
      b.setScope(MonitorElementData::Scope::JOB);
      this->dqmEndRun(b, g, run, es);
    });

    run.put(runToken_, std::make_unique<DQMToken>());
  }

  void endRun(edm::Run const &, edm::EventSetup const &) override{};

  void endJob() final {
    dqmstore_->meBookerGetter([this](DQMStore::IBooker &b, DQMStore::IGetter &g) {
      b.setScope(MonitorElementData::Scope::JOB);
      this->dqmEndJob(b, g);
    });
  };

  ~DQMEDHarvester() override = default;

  // DQM_EXPERIMENTAL
  // Could be used for niche workflows like commissioning.
  // Real harvesting jobs have no events and will never call this.
  virtual void dqmAnalyze(DQMStore::IBooker &, DQMStore::IGetter &, edm::Event const &, edm::EventSetup const &){};
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &,
                                     DQMStore::IGetter &,
                                     edm::LuminosityBlock const &,
                                     edm::EventSetup const &){};
  // HARVESTING should happen in endJob (or endLumi, for online), but there can
  // be applications for end-run harvesting. Better to have a callback than
  // have unprotected DQMStore access.
  virtual void dqmEndRun(DQMStore::IBooker &, DQMStore::IGetter &, edm::Run const &, edm::EventSetup const &){};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) = 0;
};

#endif  // CORE_DQMED_HARVESTER_H

#ifndef DQMServices_Core_DQMGlobalEDAnalyzer_h
#define DQMServices_Core_DQMGlobalEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Histograms/interface/DQMToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

template <typename H, typename... Args>
class DQMGlobalEDAnalyzerBase
    : public edm::global::EDProducer<edm::RunCache<H>,
                                     // DQMGlobalEDAnalyzer are fundamentally unable to produce histograms for any
                                     // other scope than MonitorElement::Scope::RUN.
                                     edm::EndRunProducer,
                                     edm::Accumulator,
                                     Args...> {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  // framework calls in the order of invocation
  DQMGlobalEDAnalyzerBase() {
    // for whatever reason we need the explicit `template` keyword here.
    runToken_ = this->template produces<DQMToken, edm::Transition::EndRun>("DQMGenerationRecoRun");
    dqmstore_ = edm::Service<DQMStore>().operator->();
  }

  std::shared_ptr<H> globalBeginRun(edm::Run const& run, edm::EventSetup const& setup) const final {
    auto h = std::make_shared<H>();

    dqmBeginRun(run, setup, *h);

    // in case of concurrent runs, this will create clones of the already
    // booked MEs.
    dqmstore_->bookTransaction(
        [&, this](DQMStore::IBooker& b) {
          // this runs while holding the DQMStore lock
          b.cd();
          bookHistograms(b, run, setup, *h);
        },
        // The run number is part of the module ID here, since we want distinct
        // local MEs for each run cache.
        meId(run),
        /* canSaveByLumi */ false);
    dqmstore_->initLumi(run.run(), /* lumi */ 0, meId(run));
    dqmstore_->enterLumi(run.run(), /* lumi */ 0, meId(run));
    return h;
  }

  void accumulate(edm::StreamID id, edm::Event const& event, edm::EventSetup const& setup) const final {
    auto const& h = *this->runCache(event.getRun().index());
    dqmAnalyze(event, setup, h);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const final{};

  // methods to be implemented by the user, in order of invocation
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&, H&) const {}
  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, H&) const = 0;
  // TODO: rename this analyze() for consistency.
  virtual void dqmAnalyze(edm::Event const&, edm::EventSetup const&, H const&) const = 0;

protected:
  DQMStore* dqmstore_;
  edm::EDPutTokenT<DQMToken> runToken_;
  uint64_t meId(edm::Run const& run) const { return (((uint64_t)run.run()) << 32) + this->moduleDescription().id(); }
};

// Case without RunSummaryCache
template <typename H, typename... Args>
class DQMGlobalEDAnalyzer : public DQMGlobalEDAnalyzerBase<H, Args...> {
public:
  void globalEndRunProduce(edm::Run& run, edm::EventSetup const& setup) const final {
    auto const& h = *this->runCache(run.index());
    dqmEndRun(run, setup, h);
    this->dqmstore_->leaveLumi(run.run(), /* lumi */ 0, this->meId(run));
    run.emplace(this->runToken_);
  }

  virtual void dqmEndRun(edm::Run const&, edm::EventSetup const&, H const&) const {}
};

// Case with RunSummaryCache, must be the second template argument
template <typename H,    // type for RunCache
          typename RSC,  // type for RunSummaryCache
          typename... Args>
class DQMGlobalRunSummaryEDAnalyzer : public DQMGlobalEDAnalyzerBase<H, edm::RunSummaryCache<RSC>, Args...> {
public:
  void globalEndRunProduce(edm::Run& run, edm::EventSetup const& setup, RSC const* runSummaryCache) const final {
    auto const& h = *this->runCache(run.index());
    dqmEndRun(run, setup, h, *runSummaryCache);
    this->dqmstore_->leaveLumi(run.run(), /* lumi */ 0, this->meId(run));
    run.emplace(this->runToken_);
  }

  virtual void dqmEndRun(edm::Run const&, edm::EventSetup const&, H const&, RSC const&) const {}
};

#endif  // DQMServices_Core_DQMGlobalEDAnalyzer_h

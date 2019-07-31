#ifndef DQMServices_Core_oneDQMEDAnalyzer_h
#define DQMServices_Core_oneDQMEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

/**
 * A "one" module base class that can only produce per-run histograms. This
 * allows to easily wrap non-thread-safe code and is ok'ish performance-wise,
 * since it only blocks concurrent runs, not concurrent lumis.
 * It can be combined with edm::LuminosityBlockCache to watch per-lumi things,
 * and fill per-run histograms with the results.
 */
template<typename... Args>
class DQMOneEDAnalyzer : public edm::one::EDProducer<edm::EndRunProducer,
                                              edm::one::WatchRuns,
                                              edm::Accumulator,
                                              Args...>  {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  virtual bool getCanSaveByLumi() { return false; }

  // framework calls in the order of invocation
  DQMOneEDAnalyzer() {
    // for whatever reason we need the explicit `template` keyword here.
    runToken_ = this->template produces<DQMToken, edm::Transition::EndRun>();
  }

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final {
    dqmBeginRun(run, setup);
    edm::Service<DQMStore>()->bookTransaction(
        [this, &run, &setup](DQMStore::IBooker& booker) {
          booker.cd();
          this->bookHistograms(booker, run, setup);
        },
        run.run(),
        this->moduleDescription().id(),
        this->getCanSaveByLumi());
  }

  void accumulate(edm::Event const& event, edm::EventSetup const& setup) final {
    analyze(event, setup);
  }

  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) final {
    dqmEndRun(run, setup);
    edm::Service<DQMStore>()->cloneRunHistograms(run.run(), this->moduleDescription().id());
    run.emplace<DQMToken>(runToken_);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void endRun(edm::Run const&, edm::EventSetup const&) final{};

  // methods to be implemented by the user, in order of invocation
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) = 0;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) = 0;
  virtual void dqmEndRun(edm::Run const&, edm::EventSetup const&) {}

protected:
  edm::EDPutTokenT<DQMToken> runToken_;
};

/**
 * A "one" module base class that can also watch lumi transitions and produce
 * per-lumi MEs. This should be used carefully, since it will block concurrent
 * lumisections in the entire job!
 * Combining with edm::LuminosityBlockCache is pointless and will not work
 * properly, due to the ordering of global/produce transitions.
 * It would be possible to make this concurrent lumi-able with a bit of work
 * on the DQMStore side, but the kind of modules that need this base class
 * probaby care about seeing lumisections in order anyways.
 */

template<typename... Args>
class DQMOneLumiEDAnalyzer : public DQMOneEDAnalyzer<edm::EndLuminosityBlockProducer,
                                              edm::one::WatchLuminosityBlocks,
                                              Args...>  {

  bool getCanSaveByLumi() override { return true; }

  // framework calls in the order of invocation
  DQMOneLumiEDAnalyzer() {
    // for whatever reason we need the explicit `template` keyword here.
    lumiToken_ = this->template produces<DQMToken, edm::Transition::EndLuminosityBlock>();
  }

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) final {
    dqmBeginLumi(lumi, setup);
    this->dqmstore_->enterLumi(lumi.run(), lumi.luminosityBlock());
  }

  //void accumulate(edm::StreamID id, edm::Event const& event, edm::EventSetup const& setup) final {
  //  // TODO: we could maybe switch lumis by event here, to allow concurrent
  //  // lumis. Not for now, though.
  //  analyze(event, setup);
  //}

  void endLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& setup) final {
    dqmEndLumi(lumi, setup);
    edm::Service<DQMStore>()->cloneLumiHistograms(
          lumi.run(),
          lumi.luminosityBlock(),
          moduleDescription().id());
    lumi.emplace(lumiToken_);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final{};

  // methods to be implemented by the user, in order of invocation
  virtual void dqmBeginLumi(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void dqmEndLumi(edm::LuminosityBlock const&, edm::EventSetup const&) {}

private:
  edm::EDPutTokenT<DQMToken> lumiToken_;
};

#endif  // DQMServices_Core_DQMEDAnalyzer_h

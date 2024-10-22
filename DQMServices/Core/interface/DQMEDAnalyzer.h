#ifndef DQMServices_Core_DQMEDAnalyzer_h
#define DQMServices_Core_DQMEDAnalyzer_h

#include "FWCore/Framework/interface/stream/makeGlobal.h"

struct DQMEDAnalyzerGlobalCache;

// If we declare a global cache (which is not absolutely needed right now, but
// might be in the future), the framework will try to pass it to the
// constructor. But, we don't want to change all subsystem code whenever we
// change that implementation detail, so instead we hack the framework to not
// do that. See issue #27125.
namespace edm::stream::impl {
  template <typename T>
  T* makeStreamModule(edm::ParameterSet const& iPSet, DQMEDAnalyzerGlobalCache const* global) {
    return new T(iPSet);
  }
}  // namespace edm::stream::impl

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

struct DQMEDAnalyzerGlobalCache {
  // slightly overkill for now, but we might want to putt the full DQMStore
  // here at some point.
  mutable std::mutex master_;
  mutable edm::EDPutTokenT<DQMToken> lumiToken_;
  mutable edm::EDPutTokenT<DQMToken> runToken_;
};

/**
 * The standard DQM module base.
 */
class DQMEDAnalyzer : public edm::stream::EDProducer<edm::GlobalCache<DQMEDAnalyzerGlobalCache>,
                                                     edm::EndRunProducer,
                                                     edm::EndLuminosityBlockProducer,
                                                     edm::Accumulator> {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  virtual bool getCanSaveByLumi() { return true; }

  // framework calls in the order of invocation

  static std::unique_ptr<DQMEDAnalyzerGlobalCache> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<DQMEDAnalyzerGlobalCache>();
  }

  DQMEDAnalyzer() {
    // for whatever reason we need the explicit `template` keyword here.
    runToken_ = this->template produces<DQMToken, edm::Transition::EndRun>("DQMGenerationRecoRun");
    lumiToken_ = this->template produces<DQMToken, edm::Transition::EndLuminosityBlock>("DQMGenerationRecoLumi");
    streamId_ = edm::StreamID::invalidStreamID().value();
  }

  void beginStream(edm::StreamID id) final {
    assert(streamId_ == edm::StreamID::invalidStreamID().value() || streamId_ == id.value());
    this->streamId_ = id.value();
    // now, since we can't access the global cache in the constructor (we
    // blocked that above to not expose the cache to the subsystem code,
    // we need to store the tokens here.
    // This also requires locking, since the streams will run in parallel.
    // See https://github.com/cms-sw/cmssw/issues/27291#issuecomment-505909101
    auto lock = std::scoped_lock(globalCache()->master_);
    if (globalCache()->runToken_.isUninitialized()) {
      globalCache()->lumiToken_ = lumiToken_;
      globalCache()->runToken_ = runToken_;
    }
  }

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final {
    // if we run booking multiple times because there are multiple runs in a
    // job, this is needed to make sure all existing MEs are in a valid state
    // before the booking code runs.
    edm::Service<DQMStore>()->initLumi(run.run(), /* lumi */ 0, meId());
    edm::Service<DQMStore>()->enterLumi(run.run(), /* lumi */ 0, meId());
    dqmBeginRun(run, setup);
    edm::Service<DQMStore>()->bookTransaction(
        [this, &run, &setup](DQMStore::IBooker& booker) {
          booker.cd();
          this->bookHistograms(booker, run, setup);
        },
        meId(),
        this->getCanSaveByLumi());
    edm::Service<DQMStore>()->initLumi(run.run(), /* lumi */ 0, meId());
    edm::Service<DQMStore>()->enterLumi(run.run(), /* lumi */ 0, meId());
  }

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) final {
    edm::Service<DQMStore>()->initLumi(lumi.run(), lumi.luminosityBlock(), meId());
    edm::Service<DQMStore>()->enterLumi(lumi.run(), lumi.luminosityBlock(), meId());
  }

  void accumulate(edm::Event const& event, edm::EventSetup const& setup) final { analyze(event, setup); }

  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) final {
    edm::Service<DQMStore>()->leaveLumi(lumi.run(), lumi.luminosityBlock(), meId());
  }

  static void globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumi,
                                              edm::EventSetup const& setup,
                                              LuminosityBlockContext const* context) {
    lumi.emplace(context->global()->lumiToken_);
  }

  void endRun(edm::Run const& run, edm::EventSetup const& setup) final {
    edm::Service<DQMStore>()->leaveLumi(run.run(), /* lumi */ 0, meId());
  }
  static void globalEndRunProduce(edm::Run& run, edm::EventSetup const& setup, RunContext const* context) {
    run.emplace<DQMToken>(context->global()->runToken_);
  }

  static void globalEndJob(DQMEDAnalyzerGlobalCache const*) {}

  // methods to be implemented by the user, in order of invocation
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) = 0;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) {}

protected:
  edm::EDPutTokenT<DQMToken> runToken_;
  edm::EDPutTokenT<DQMToken> lumiToken_;
  unsigned int streamId_;
  uint64_t meId() const { return (((uint64_t)streamId_) << 32) + this->moduleDescription().id(); }
};

#endif  // DQMServices_Core_DQMEDAnalyzer_h

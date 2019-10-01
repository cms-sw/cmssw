#ifndef DQMServices_Core_DQMEDAnalyzer_h
#define DQMServices_Core_DQMEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

/**
 * The standard DQM module base. For now, this is the same as DQMOneLumiEDAnalyzer,
 * but they can (and will) diverge in the future.
 */
class DQMEDAnalyzer : public edm::one::EDProducer<edm::EndRunProducer,
                                                  edm::one::WatchRuns,
                                                  edm::EndLuminosityBlockProducer,
                                                  edm::one::WatchLuminosityBlocks,
                                                  edm::Accumulator> {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  virtual bool getCanSaveByLumi() { return true; }

  // framework calls in the order of invocation
  DQMEDAnalyzer() {
    // for whatever reason we need the explicit `template` keyword here.
    runToken_ = this->template produces<DQMToken, edm::Transition::EndRun>();
    lumiToken_ = this->template produces<DQMToken, edm::Transition::EndLuminosityBlock>();
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

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) final {
    dqmBeginLuminosityBlock(lumi, setup);
  }

  void accumulate(edm::Event const& event, edm::EventSetup const& setup) final { analyze(event, setup); }

  void endLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& setup) final {
    dqmEndLuminosityBlock(lumi, setup);
    edm::Service<DQMStore>()->cloneLumiHistograms(lumi.run(), lumi.luminosityBlock(), this->moduleDescription().id());
    lumi.emplace(lumiToken_);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final{};

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
  virtual void dqmBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) {}
  virtual void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void dqmEndRun(edm::Run const&, edm::EventSetup const&) {}

protected:
  edm::EDPutTokenT<DQMToken> runToken_;
  edm::EDPutTokenT<DQMToken> lumiToken_;
};

#endif  // DQMServices_Core_DQMEDAnalyzer_h

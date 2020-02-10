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
 * The standard DQM module base.
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
    runToken_ = this->template produces<DQMToken, edm::Transition::EndRun>("DQMGenerationRecoRun");
    lumiToken_ = this->template produces<DQMToken, edm::Transition::EndLuminosityBlock>("DQMGenerationRecoLumi");
  }

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final {
    dqmBeginRun(run, setup);
    edm::Service<DQMStore>()->bookTransaction(
        [this, &run, &setup](DQMStore::IBooker& booker) {
          booker.cd();
          this->bookHistograms(booker, run, setup);
        },
        this->moduleDescription().id(),
        this->getCanSaveByLumi());
    edm::Service<DQMStore>()->enterLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
  }

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) final {
    edm::Service<DQMStore>()->enterLumi(lumi.run(), lumi.luminosityBlock(), this->moduleDescription().id());
  }

  void accumulate(edm::Event const& event, edm::EventSetup const& setup) final { analyze(event, setup); }

  void endLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& setup) final {
    edm::Service<DQMStore>()->leaveLumi(lumi.run(), lumi.luminosityBlock(), this->moduleDescription().id());
    lumi.emplace(lumiToken_);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) final{};

  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) final {
    edm::Service<DQMStore>()->leaveLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
    run.emplace<DQMToken>(runToken_);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void endRun(edm::Run const&, edm::EventSetup const&) final{};

  // methods to be implemented by the user, in order of invocation
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) = 0;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) {}

protected:
  edm::EDPutTokenT<DQMToken> runToken_;
  edm::EDPutTokenT<DQMToken> lumiToken_;
};

#endif  // DQMServices_Core_DQMEDAnalyzer_h

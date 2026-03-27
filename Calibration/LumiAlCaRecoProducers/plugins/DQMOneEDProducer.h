#ifndef DQMServices_Core_DQMOneEDProducer_h
#define DQMServices_Core_DQMOneEDProducer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

template <typename... Args>
class DQMOneEDProducer
    : public edm::one::EDProducer<edm::EndRunProducer, edm::one::WatchRuns, edm::Accumulator, Args...> {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  virtual bool getCanSaveByLumi() { return true; }

  // framework calls in the order of invocation
  DQMOneEDProducer() {
    // for whatever reason we need the explicit `template` keyword here.
    runToken_ = this->template produces<DQMToken, edm::Transition::EndRun>("DQMGenerationRecoRun");
  }

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final {
    // if we run booking multiple times because there are multiple runs in a
    // job, this is needed to make sure all existing MEs are in a valid state
    // before the booking code runs.
    edm::Service<DQMStore>()->initLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
    edm::Service<DQMStore>()->enterLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
    dqmBeginRun(run, setup);
    edm::Service<DQMStore>()->bookTransaction(
        [this, &run, &setup](DQMStore::IBooker& booker) {
          booker.cd();
          this->bookHistograms(booker, run, setup);
        },
        this->moduleDescription().id(),
        this->getCanSaveByLumi());
    edm::Service<DQMStore>()->initLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
    edm::Service<DQMStore>()->enterLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
  }

  void accumulate(edm::Event const& event, edm::EventSetup const& setup) override {
    auto& lumi = event.getLuminosityBlock();
    edm::Service<dqm::legacy::DQMStore>()->enterLumi(
        lumi.run(), lumi.luminosityBlock(), this->moduleDescription().id());
    analyze(event, setup);
    edm::Service<dqm::legacy::DQMStore>()->leaveLumi(
        lumi.run(), lumi.luminosityBlock(), this->moduleDescription().id());
  }

  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) final {
    dqmEndRun(run, setup);
    edm::Service<DQMStore>()->leaveLumi(run.run(), /* lumi */ 0, this->moduleDescription().id());
    run.emplace<DQMToken>(runToken_);
  }

  // Subsystems could safely override this, but any changes to MEs would not be
  // noticeable since the product was made already.
  void endRun(edm::Run const&, edm::EventSetup const&) final {}

protected:
  // methods to be implemented by the user, in order of invocation
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) = 0;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) {}
  virtual void dqmEndRun(edm::Run&, edm::EventSetup const&) {}

  edm::EDPutTokenT<DQMToken> runToken_;
};

#endif  // DQMServices_Core_DQMOneEDProducer_h

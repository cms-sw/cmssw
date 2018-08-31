#ifndef DQMServices_Core_oneDQMEDAnalyzer_h
#define DQMServices_Core_oneDQMEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

/*
A one::DQMEDAnalyzer<*> should be used in the case where a module must file DQM histograms and must be an 
edm::one module. 

Inheriting from one::DQMEDAnalyzer<> gives access to Run transitions.

Inheriting from one::DQMEDAnalyzer<edm::one::WatchLuminosityBlocks> gives access to Run and LuminosityBlock transitions but can only be used with Run based MonitorElements.

Inheriting from one::DQMEDAnalyzer<one::DQMLuminosityBlockElements> give access to Run and LuminosityBlock transitions and filling LuminosityBlock based MonitorElements.
*/

namespace one {

struct DQMLuminosityBlockElements {};

namespace dqmimplementation {
template <typename... T>
class DQMRunEDProducer : public edm::one::EDProducer<edm::Accumulator,
                                                     edm::EndRunProducer,
                                                     edm::one::WatchRuns, T...> 
{
public:
  DQMRunEDProducer() :
    runToken_{this-> template produces<DQMToken,edm::Transition::EndRun>("endRun")}
    {}
  ~DQMRunEDProducer() override = default;
  DQMRunEDProducer(DQMRunEDProducer<T...> const&) = delete;
  DQMRunEDProducer(DQMRunEDProducer<T...> &&) = delete;

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final {
    dqmBeginRun(run, setup);
    edm::Service<DQMStore>()->bookTransaction(
    [this, &run, &setup](DQMStore::IBooker & booker)
    {
      booker.cd();
      this->bookHistograms(booker, run, setup);
    },
    run.run(),
    this->moduleDescription().id());
  }

  void endRun(edm::Run const& run, edm::EventSetup const& setup) override {}
  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) override {
    edm::Service<DQMStore>()->cloneRunHistograms(
        run.run(),
        this->moduleDescription().id());

    run.emplace<DQMToken>(runToken_);
  }

  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;

  virtual void analyze(edm::Event const&, edm::EventSetup const&) {}
  void accumulate(edm::Event const& ev, edm::EventSetup const& es) final {
    analyze(ev,es);
  }
private:
  edm::EDPutTokenT<DQMToken> runToken_;

};

class DQMLumisEDProducer : public DQMRunEDProducer<edm::EndLuminosityBlockProducer,
                                                   edm::one::WatchLuminosityBlocks>

{
public:
  DQMLumisEDProducer();
  ~DQMLumisEDProducer() override = default;
  DQMLumisEDProducer(DQMLumisEDProducer const&) = delete;
  DQMLumisEDProducer(DQMLumisEDProducer &&) = delete;

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;

  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void endLuminosityBlockProduce(edm::LuminosityBlock & lumi, edm::EventSetup const& setup) final;


  virtual void dqmBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  //virtual void bookLumiHistograms(DQMStore::IBooker &i, edm::LuminosityBlock const&, edm::EventSetup const&) = 0;

 private:
  edm::EDPutTokenT<DQMToken> lumiToken_;
};

template <typename... T> class DQMBaseClass;

template<> class DQMBaseClass<> : public DQMRunEDProducer<> {};
template<> class DQMBaseClass<DQMLuminosityBlockElements> : public DQMLumisEDProducer {};
template<> class DQMBaseClass<edm::one::WatchLuminosityBlocks> : public DQMRunEDProducer<edm::one::WatchLuminosityBlocks> {};
}

template <typename... T>
class DQMEDAnalyzer : public dqmimplementation::DQMBaseClass<T...>
{
public:
  DQMEDAnalyzer() = default;
  ~DQMEDAnalyzer() override = default;
  DQMEDAnalyzer(DQMEDAnalyzer<T...> const&) = delete;
  DQMEDAnalyzer(DQMEDAnalyzer<T...> &&) = delete;
};
}
#endif // DQMServices_Core_DQMEDAnalyzer_h

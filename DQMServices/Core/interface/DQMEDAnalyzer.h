#ifndef DQMServices_Core_DQMEDAnalyzer_h
#define DQMServices_Core_DQMEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class DQMEDAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks>
{
public:
  DQMEDAnalyzer() = default;
  ~DQMEDAnalyzer() override = default;
  DQMEDAnalyzer(DQMEDAnalyzer const&) = delete;
  DQMEDAnalyzer(DQMEDAnalyzer &&) = delete;

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final
  {
    dqmBeginRun(run, setup);
    edm::Service<DQMStore>()->bookTransaction(
      [this, &run, &setup](DQMStore::IBooker & booker)
      {
        booker.cd();
        this->bookHistograms(booker, run, setup);
      },
      run.run(),
      run.moduleCallingContext()->moduleDescription()->id());
  }

  void endRun(edm::Run const& run, edm::EventSetup const& setup) override
  { }

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override
  { }

  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override
  { }

  /* uncomment this after the migration to
   * edm::one::EDProducer<edm::Accumulator,
   *                      edm::one::WatchLuminosityBlocks,
   *                      edm::EndLuminosityBlockProducer,
   *                      edm::one::WatchRuns,
   *                      edm::EndRunProducer>
   *
  void endLuminosityBlockProduce(edm::LuminosityBlock & lumi, edm::EventSetup const& setup) final
  {
    edm::Service<DQMStore>()->cloneLumiHistograms(
        lumi.run(),
        lumi.luminosityBlock(),
        lumi.moduleCallingContext()->moduleDescription()->id());
  }
  */

  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;
};

#endif // DQMServices_Core_DQMEDAnalyzer_h

#ifndef DQMServices_Core_DQMEDAnalyzer_h
#define DQMServices_Core_DQMEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class DQMEDAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks>
{
public:
  DQMEDAnalyzer() = default;
  ~DQMEDAnalyzer() = default;
  DQMEDAnalyzer(DQMEDAnalyzer const&) = delete;
  DQMEDAnalyzer(DQMEDAnalyzer &&) = delete;

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final
  {
    dqmBeginRun(run, setup);
    DQMStore * store = edm::Service<DQMStore>().operator->();
    store->bookTransaction(
      [this, &run, &setup](DQMStore::IBooker & booker)
      {
        booker.cd();
        this->bookHistograms(booker, run, setup);
      },
      run.run());
  }

  void endRun(edm::Run const& run, edm::EventSetup const& setup)
  { }

  void beginLuminosityBlock(edm::LuminosityBlock const& run, edm::EventSetup const& setup)
  { }

  void endLuminosityBlock(edm::LuminosityBlock const& run, edm::EventSetup const& setup)
  { }

  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&) {}
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;
};

#endif // DQMServices_Core_DQMEDAnalyzer_h

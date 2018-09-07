#ifndef DQMServices_Core_DQMGlobalEDAnalyzer_h
#define DQMServices_Core_DQMGlobalEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

template <typename H, typename... Args>
class DQMGlobalEDAnalyzer : public edm::global::EDAnalyzer<edm::RunCache<H>, Args...>
{
private:
  std::shared_ptr<H>
  globalBeginRun(edm::Run const&, edm::EventSetup const&) const final;

  void
  globalEndRun(edm::Run const&, edm::EventSetup const&) const final;

  virtual void
  dqmBeginRun(edm::Run const&, edm::EventSetup const&, H &) const { }

  // this will run while holding the DQMStore lock
  virtual void
  bookHistograms(DQMStore::ConcurrentBooker &, edm::Run const&, edm::EventSetup const&, H &) const = 0;

  void
  analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final;

  virtual void
  dqmAnalyze(edm::Event const&, edm::EventSetup const&, H const&) const = 0;
};

template <typename H, typename... Args>
std::shared_ptr<H>
DQMGlobalEDAnalyzer<H, Args...>::globalBeginRun(edm::Run const& run, edm::EventSetup const& setup) const
{
  auto h = std::make_shared<H>();
  dqmBeginRun(run, setup, *h);
  edm::Service<DQMStore>()->bookConcurrentTransaction([&, this](DQMStore::ConcurrentBooker &b) {
      // this runs while holding the DQMStore lock
      b.cd();
      bookHistograms(b, run, setup, *h);
    },
    run.run() );
  return h;
}

template <typename H, typename... Args>
void
DQMGlobalEDAnalyzer<H, Args...>::globalEndRun(edm::Run const&, edm::EventSetup const&) const
{
}

template <typename H, typename... Args>
void
DQMGlobalEDAnalyzer<H, Args...>::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const
{
  //auto& h = const_cast<H&>(* this->runCache(event.getRun().index()));
  auto const& h = * this->runCache(event.getRun().index());
  dqmAnalyze(event, setup, h);
}

#endif // DQMServices_Core_DQMGlobalEDAnalyzer_h

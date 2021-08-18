#ifndef DQM_HcalTasks_PedestalRunSummary_h
#define DQM_HcalTasks_PedestalRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm {
  class PedestalRunSummary : public DQClient {
  public:
    PedestalRunSummary(std::string const &, std::string const &, edm::ParameterSet const &, edm::ConsumesCollector &iC);
    ~PedestalRunSummary() override {}

    void beginRun(edm::Run const &, edm::EventSetup const &) override;
    void endLuminosityBlock(DQMStore::IBooker &,
                            DQMStore::IGetter &,
                            edm::LuminosityBlock const &,
                            edm::EventSetup const &) override;
    std::vector<flag::Flag> endJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  protected:
  };
}  // namespace hcaldqm

#endif

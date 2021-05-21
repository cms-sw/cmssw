#include "DQM/HcalTasks/interface/PedestalRunSummary.h"

namespace hcaldqm {
  PedestalRunSummary::PedestalRunSummary(std::string const& name,
                                         std::string const& taskname,
                                         edm::ParameterSet const& ps,
                                         edm::ConsumesCollector& iC)
      : DQClient(name, taskname, ps, iC) {}

  /* virtual */ void PedestalRunSummary::beginRun(edm::Run const& r, edm::EventSetup const& es) {
    DQClient::beginRun(r, es);
  }

  /* virtual */ void PedestalRunSummary::endLuminosityBlock(DQMStore::IBooker& ib,
                                                            DQMStore::IGetter& ig,
                                                            edm::LuminosityBlock const& lb,
                                                            edm::EventSetup const& es) {
    DQClient::endLuminosityBlock(ib, ig, lb, es);
  }

  /* virtual */ std::vector<flag::Flag> PedestalRunSummary::endJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
    std::vector<flag::Flag> sumflags;
    return sumflags;
  }
}  // namespace hcaldqm

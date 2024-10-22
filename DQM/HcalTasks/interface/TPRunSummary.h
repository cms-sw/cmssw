#ifndef DQM_HcalTasks_TPRunSummary_h
#define DQM_HcalTasks_TPRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm {
  class TPRunSummary : public DQClient {
  public:
    TPRunSummary(std::string const &, std::string const &, edm::ParameterSet const &, edm::ConsumesCollector &iC);
    ~TPRunSummary() override {}

    void beginRun(edm::Run const &, edm::EventSetup const &) override;
    void endLuminosityBlock(DQMStore::IBooker &,
                            DQMStore::IGetter &,
                            edm::LuminosityBlock const &,
                            edm::EventSetup const &) override;
    std::vector<flag::Flag> endJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  protected:
    ContainerSingle2D _cEtMsmFraction_depthlike;
    ContainerSingle2D _cFGMsmFraction_depthlike;

    double _thresh_FGMsmRate_high, _thresh_FGMsmRate_low;
    double _thresh_EtMsmRate_high, _thresh_EtMsmRate_low;

    enum TPFlag { fEtMsm = 0, fFGMsm = 1, nTPFlag = 3 };
  };
}  // namespace hcaldqm

#endif

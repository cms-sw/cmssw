#ifndef DQM_HcalTasks_DigiRunSummary_h
#define DQM_HcalTasks_DigiRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

namespace hcaldqm {
  class DigiRunSummary : public DQClient {
  public:
    DigiRunSummary(std::string const &, std::string const &, edm::ParameterSet const &, edm::ConsumesCollector &iC);
    ~DigiRunSummary() override {}

    void beginRun(edm::Run const &, edm::EventSetup const &) override;
    void endLuminosityBlock(DQMStore::IBooker &,
                            DQMStore::IGetter &,
                            edm::LuminosityBlock const &,
                            edm::EventSetup const &) override;
    std::vector<flag::Flag> endJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  protected:
    std::vector<LSSummary> _vflagsLS;

    double _thresh_unihf;

    electronicsmap::ElectronicsMap _ehashmap;

    std::vector<uint32_t> _vhashVME, _vhashuTCA, _vhashFEDHF;
    std::vector<int> _vFEDsVME, _vFEDsuTCA;
    filter::HashFilter _filter_VME, _filter_uTCA, _filter_FEDHF;

    Container2D _cOccupancy_depth;
    bool _booked;
    MonitorElement *_meNumEvents;  // number of events vs LS

    ContainerXXX<uint32_t> _xDead, _xDigiSize, _xUniHF, _xUni, _xNChs, _xNChsNominal;

    std::map<HcalSubdetector, uint32_t> _refDigiSize;

    //	flag enum
    enum DigiLSFlag {
      fDigiSize = 0,
      fNChsHF = 1,
      fUnknownIds = 2,
      fLED = 3,
      nLSFlags = 4,  // defines the boundary between lumi-based and run-based flags
      fUniHF = 5,
      fDead = 6,
      nDigiFlag = 7
    };
  };
}  // namespace hcaldqm

#endif

#ifndef DQM_HcalTasks_NoCQTask_h
#define DQM_HcalTasks_NoCQTask_h

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"

class NoCQTask : public hcaldqm::DQTask {
public:
  NoCQTask(edm::ParameterSet const &);
  ~NoCQTask() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  std::shared_ptr<hcaldqm::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const &,
                                                             edm::EventSetup const &) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

protected:
  void _process(edm::Event const &, edm::EventSetup const &) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  edm::InputTag _tagHBHE;
  edm::InputTag _tagHO;
  edm::InputTag _tagHF;
  edm::InputTag _tagReport;
  edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
  edm::EDGetTokenT<HODigiCollection> _tokHO;
  edm::EDGetTokenT<HFDigiCollection> _tokHF;
  edm::EDGetTokenT<HcalUnpackerReport> _tokReport;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  double _cutSumQ_HBHE, _cutSumQ_HO, _cutSumQ_HF;

  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

  hcaldqm::ContainerProf2D _cTimingCut_depth;
  hcaldqm::Container2D _cOccupancy_depth;
  hcaldqm::Container2D _cOccupancyCut_depth;
  hcaldqm::Container2D _cBadQuality_depth;
};

#endif

#ifndef TPComparisonTask_h
#define TPComparisonTask_h

/**
 *	file:		TPComparisonTask.h
 *	Author:		Viktor Khristenko
 *	Date:		08.12.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class TPComparisonTask : public hcaldqm::DQTask {
public:
  TPComparisonTask(edm::ParameterSet const&);
  ~TPComparisonTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

protected:
  //	funcs
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  //	Tags and corresponding Tokens
  edm::InputTag _tag1;
  edm::InputTag _tag2;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tok1;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tok2;

  //	tmp flags
  bool _skip1x1;

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmapuTCA;
  hcaldqm::electronicsmap::ElectronicsMap _ehashmapVME;

  //	hahses/FED vectors
  std::vector<uint32_t> _vhashFEDs;

  //	Filters
  hcaldqm::filter::HashFilter _filter_VME;
  hcaldqm::filter::HashFilter _filter_uTCA;

  /**
		 *	hcaldqm::Containers
		 */

  //	Et
  hcaldqm::Container2D _cEt_TTSubdet[4];
  hcaldqm::Container2D _cEtall_TTSubdet;

  //	FG
  hcaldqm::Container2D _cFG_TTSubdet[4];

  //	Missing
  hcaldqm::Container2D _cMsn_FEDVME;
  hcaldqm::Container2D _cMsn_FEDuTCA;
  hcaldqm::ContainerSingle2D _cMsnVME;
  hcaldqm::ContainerSingle2D _cMsnuTCA;

  //	mismatches
  hcaldqm::Container2D _cEtMsm_FEDVME;
  hcaldqm::Container2D _cEtMsm_FEDuTCA;
  hcaldqm::Container2D _cFGMsm_FEDVME;
  hcaldqm::Container2D _cFGMsm_FEDuTCA;

  //	depth like
  hcaldqm::ContainerSingle2D _cEtMsm;
  hcaldqm::ContainerSingle2D _cFGMsm;
};

#endif

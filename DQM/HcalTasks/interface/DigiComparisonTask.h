#ifndef DigiComparisonTask_h
#define DigiComparisonTask_h

/**
 *	file:		DigiComparisonTask.h
 *	Author:		Viktor Khristenko
 *	Date:		08.12.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/HashFilter.h"

class DigiComparisonTask : public hcaldqm::DQTask {
public:
  DigiComparisonTask(edm::ParameterSet const&);
  ~DigiComparisonTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

protected:
  //	funcs
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  //	Tags and corresponding Tokens
  edm::InputTag _tagHBHE1;
  edm::InputTag _tagHBHE2;
  edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE1;
  edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE2;

  //	emap+hashmap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmapuTCA;
  hcaldqm::electronicsmap::ElectronicsMap _ehashmapVME;

  //	hashes/FED vectors
  std::vector<uint32_t> _vhashFEDs;

  //	Filters
  hcaldqm::filter::HashFilter _filter_VME;
  hcaldqm::filter::HashFilter _filter_uTCA;

  /**
		 *	Containers
		 */

  //	ADC
  hcaldqm::Container2D _cADC_Subdet[10];
  hcaldqm::Container2D _cADCall_Subdet;

  //	Mismatched
  hcaldqm::Container2D _cMsm_FEDVME;
  hcaldqm::Container2D _cMsm_FEDuTCA;
  hcaldqm::Container2D _cMsm_depth;

  //	Missing Completely
  hcaldqm::Container1D _cADCMsnuTCA_Subdet;
  hcaldqm::Container1D _cADCMsnVME_Subdet;
  hcaldqm::Container2D _cMsnVME_depth;
  hcaldqm::Container2D _cMsnuTCA_depth;
  hcaldqm::Container2D _cMsn_FEDVME;
  hcaldqm::Container2D _cMsn_FEDuTCA;
};

#endif

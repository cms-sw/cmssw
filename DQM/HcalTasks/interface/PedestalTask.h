#ifndef PedestalTask_h
#define PedestalTask_h

/*
 *	file:			PedestalTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class PedestalTask : public hcaldqm::DQTask {
public:
  PedestalTask(edm::ParameterSet const&);
  ~PedestalTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void dqmEndRun(edm::Run const&, edm::EventSetup const&) override;

protected:
  //	funcs
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;
  bool _isApplicable(edm::Event const&) override;
  virtual void _dump();

  //	tags and tokens
  edm::InputTag _tagQIE11;
  edm::InputTag _tagHO;
  edm::InputTag _tagQIE10;
  edm::InputTag _tagTrigger;
  edm::InputTag _taguMN;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;
  edm::EDGetTokenT<QIE11DigiCollection> _tokQIE11;
  edm::EDGetTokenT<HODigiCollection> _tokHO;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
  edm::EDGetTokenT<HcalTBTriggerData> _tokTrigger;

  std::vector<hcaldqm::flag::Flag> _vflags;
  enum PedestalFlag { fMsn = 0, fBadM = 1, fBadR = 2, nPedestalFlag = 3 };

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
  hcaldqm::filter::HashFilter _filter_uTCA;
  hcaldqm::filter::HashFilter _filter_VME;
  hcaldqm::filter::HashFilter _filter_C38;

  //	thresholds
  double _thresh_mean, _thresh_rms, _thresh_badm, _thresh_badr;
  double _thresh_missing_high, _thresh_missing_low;

  //	hashed ids of FEDs
  std::vector<uint32_t> _vhashFEDs;

  //	need containers total over the run and per 1LS
  hcaldqm::ContainerXXX<double> _xPedSum1LS;
  hcaldqm::ContainerXXX<double> _xPedSum21LS;
  hcaldqm::ContainerXXX<int> _xPedEntries1LS;
  hcaldqm::ContainerXXX<double> _xPedSumTotal;
  hcaldqm::ContainerXXX<double> _xPedSum2Total;
  hcaldqm::ContainerXXX<int> _xPedEntriesTotal;
  hcaldqm::ContainerXXX<int> _xNChs;     // number of channels per FED as in emap
  hcaldqm::ContainerXXX<int> _xNMsn1LS;  // #missing for 1LS per FED
  hcaldqm::ContainerXXX<int> _xNBadMean1LS, _xNBadRMS1LS;

  //	CondBD Reference
  hcaldqm::ContainerXXX<double> _xPedRefMean;
  hcaldqm::ContainerXXX<double> _xPedRefRMS;

  //	1D actual Means/RMSs
  hcaldqm::Container1D _cMeanTotal_Subdet;
  hcaldqm::Container1D _cRMSTotal_Subdet;
  hcaldqm::Container1D _cMean1LS_Subdet;  // 1LS
  hcaldqm::Container1D _cRMS1LS_Subdet;   // 1LS

  //	2D actual values
  hcaldqm::ContainerProf2D _cMean1LS_depth;    // 1LS
  hcaldqm::ContainerProf2D _cRMS1LS_depth;     //  1lS
  hcaldqm::ContainerProf2D _cMean1LS_FEDVME;   // 1ls
  hcaldqm::ContainerProf2D _cMean1LS_FEDuTCA;  // 1ls
  hcaldqm::ContainerProf2D _cRMS1LS_FEDVME;    // 1ls
  hcaldqm::ContainerProf2D _cRMS1LS_FEDuTCA;   // 1ls

  hcaldqm::ContainerProf2D _cMeanTotal_depth;
  hcaldqm::ContainerProf2D _cRMSTotal_depth;
  hcaldqm::ContainerProf2D _cMeanTotal_FEDVME;
  hcaldqm::ContainerProf2D _cMeanTotal_FEDuTCA;
  hcaldqm::ContainerProf2D _cRMSTotal_FEDVME;
  hcaldqm::ContainerProf2D _cRMSTotal_FEDuTCA;

  //	Comparison with DB Conditions
  hcaldqm::Container1D _cMeanDBRef1LS_Subdet;  // 1LS
  hcaldqm::Container1D _cRMSDBRef1LS_Subdet;   // 1LS
  hcaldqm::Container1D _cMeanDBRefTotal_Subdet;
  hcaldqm::Container1D _cRMSDBRefTotal_Subdet;
  hcaldqm::ContainerProf2D _cMeanDBRef1LS_depth;
  hcaldqm::ContainerProf2D _cRMSDBRef1LS_depth;
  hcaldqm::ContainerProf2D _cMeanDBRef1LS_FEDVME;
  hcaldqm::ContainerProf2D _cMeanDBRef1LS_FEDuTCA;
  hcaldqm::ContainerProf2D _cRMSDBRef1LS_FEDVME;
  hcaldqm::ContainerProf2D _cRMSDBRef1LS_FEDuTCA;

  hcaldqm::ContainerProf2D _cMeanDBRefTotal_depth;
  hcaldqm::ContainerProf2D _cRMSDBRefTotal_depth;
  hcaldqm::ContainerProf2D _cMeanDBRefTotal_FEDVME;
  hcaldqm::ContainerProf2D _cMeanDBRefTotal_FEDuTCA;
  hcaldqm::ContainerProf2D _cRMSDBRefTotal_FEDVME;
  hcaldqm::ContainerProf2D _cRMSDBRefTotal_FEDuTCA;

  //	vs LS
  hcaldqm::Container1D _cMissingvsLS_Subdet;
  hcaldqm::Container1D _cOccupancyvsLS_Subdet;
  hcaldqm::Container1D _cNBadMeanvsLS_Subdet;
  hcaldqm::Container1D _cNBadRMSvsLS_Subdet;

  //	averaging per event
  hcaldqm::ContainerProf1D _cOccupancyEAvsLS_Subdet;

  //	map of missing channels
  hcaldqm::Container2D _cMissing1LS_depth;
  hcaldqm::Container2D _cMissing1LS_FEDVME;
  hcaldqm::Container2D _cMissing1LS_FEDuTCA;
  hcaldqm::Container2D _cMissingTotal_depth;
  hcaldqm::Container2D _cMissingTotal_FEDVME;
  hcaldqm::Container2D _cMissingTotal_FEDuTCA;

  //	Mean/RMS Bad Maps
  hcaldqm::Container2D _cMeanBad1LS_depth;
  hcaldqm::Container2D _cRMSBad1LS_depth;
  hcaldqm::Container2D _cMeanBad1LS_FEDVME;
  hcaldqm::Container2D _cRMSBad1LS_FEDuTCA;
  hcaldqm::Container2D _cRMSBad1LS_FEDVME;
  hcaldqm::Container2D _cMeanBad1LS_FEDuTCA;

  hcaldqm::Container2D _cMeanBadTotal_depth;
  hcaldqm::Container2D _cRMSBadTotal_depth;
  hcaldqm::Container2D _cMeanBadTotal_FEDVME;
  hcaldqm::Container2D _cRMSBadTotal_FEDuTCA;
  hcaldqm::Container2D _cRMSBadTotal_FEDVME;
  hcaldqm::Container2D _cMeanBadTotal_FEDuTCA;

  hcaldqm::Container1D _cADC_SubdetPM;

  //	Summaries
  hcaldqm::Container2D _cSummaryvsLS_FED;
  hcaldqm::ContainerSingle2D _cSummaryvsLS;
};

#endif

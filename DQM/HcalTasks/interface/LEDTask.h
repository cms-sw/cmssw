#ifndef LEDTask_h
#define LEDTask_h

/*
 *	file:			LEDTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "FWCore/Framework/interface/Run.h"

class LEDTask : public hcaldqm::DQTask {
public:
  LEDTask(edm::ParameterSet const&);
  ~LEDTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndRun(edm::Run const& r, edm::EventSetup const&) override {
    if (_ptype == hcaldqm::fLocal)
      if (r.runAuxiliary().run() == 1)
        return;
    this->_dump();
  }

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
  edm::EDGetTokenT<QIE11DigiCollection> _tokQIE11;
  edm::EDGetTokenT<HODigiCollection> _tokHO;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
  edm::EDGetTokenT<HcalTBTriggerData> _tokTrigger;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
  hcaldqm::filter::HashFilter _filter_uTCA;
  hcaldqm::filter::HashFilter _filter_VME;

  //	Cuts
  int _nevents;
  double _lowHBHE;
  double _lowHO;
  double _lowHF;

  //	Compact
  hcaldqm::ContainerXXX<double> _xSignalSum;
  hcaldqm::ContainerXXX<double> _xSignalSum2;
  hcaldqm::ContainerXXX<int> _xEntries;
  hcaldqm::ContainerXXX<double> _xTimingSum;
  hcaldqm::ContainerXXX<double> _xTimingSum2;

  //	1D
  hcaldqm::Container1D _cSignalMean_Subdet;
  hcaldqm::Container1D _cSignalRMS_Subdet;
  hcaldqm::Container1D _cTimingMean_Subdet;
  hcaldqm::Container1D _cTimingRMS_Subdet;

  //	Prof1D
  hcaldqm::ContainerProf1D _cShapeCut_FEDSlot;

  //	2D timing/signals
  hcaldqm::ContainerProf2D _cSignalMean_depth;
  hcaldqm::ContainerProf2D _cSignalRMS_depth;
  hcaldqm::ContainerProf2D _cTimingMean_depth;
  hcaldqm::ContainerProf2D _cTimingRMS_depth;

  hcaldqm::ContainerProf2D _cSignalMean_FEDVME;
  hcaldqm::ContainerProf2D _cSignalMean_FEDuTCA;
  hcaldqm::ContainerProf2D _cTimingMean_FEDVME;
  hcaldqm::ContainerProf2D _cTimingMean_FEDuTCA;
  hcaldqm::ContainerProf2D _cSignalRMS_FEDVME;
  hcaldqm::ContainerProf2D _cSignalRMS_FEDuTCA;
  hcaldqm::ContainerProf2D _cTimingRMS_FEDVME;
  hcaldqm::ContainerProf2D _cTimingRMS_FEDuTCA;

  //	Bad Quality and Missing Channels
  hcaldqm::Container2D _cMissing_depth;
  hcaldqm::Container2D _cMissing_FEDVME;
  hcaldqm::Container2D _cMissing_FEDuTCA;

  // For hcalcalib online LED
  hcaldqm::Container2D _cADCvsTS_SubdetPM;
  hcaldqm::Container1D _cSumQ_SubdetPM;
  hcaldqm::Container1D _cTDCTime_SubdetPM;
  hcaldqm::ContainerProf2D _cTDCTime_depth;
  hcaldqm::ContainerSingle2D _cLowSignal_CrateSlot;

  // For monitoring LED firing: ADC vs BX
  std::map<HcalSubdetector, std::vector<HcalDetId> > _ledCalibrationChannels;
  hcaldqm::Container2D _LED_ADCvsBX_Subdet;   // Pin diode amplitude vs BX for online DQM
  hcaldqm::Container2D _LED_ADCvsEvn_Subdet;  // Pin diode amplitude vs Evn for local DQM
};

#endif

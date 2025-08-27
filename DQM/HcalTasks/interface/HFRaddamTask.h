#ifndef DQM_HcalTasks_HFRaddamTask_h
#define DQM_HcalTasks_HFRaddamTask_h

/*
 *	file:			RadDamTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"

class HFRaddamTask : public hcaldqm::DQTask {
public:
  HFRaddamTask(edm::ParameterSet const&);
  ~HFRaddamTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  //	funcs
  void _process(edm::Event const&, edm::EventSetup const&) override;
  bool _isApplicable(edm::Event const&) override;

  //	Tags and Tokens
  edm::InputTag _tagHF;
  edm::InputTag _taguMN;
  edm::EDGetTokenT<QIE10DigiCollection> _tokHF;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

  edm::InputTag _tagFEDs;
  edm::EDGetTokenT<FEDRawDataCollection> _tokFEDs;

  uint32_t _laserType;
  int _nevents;

  //	vector of Detector Ids for RadDam
  std::vector<HcalDetId> _vDetIds;

  //	Cuts

  //	Compact

  //	1D
  std::vector<hcaldqm::ContainerSingle1D> _vcShape;

  // For monitoring CU Raddam firing: ADC vs TS
  std::map<HcalSubdetector, std::vector<HcalDetId> > _raddamCalibrationChannels;
  hcaldqm::ContainerSingle2D _Raddam_ADCvsTS;   // Raddam amplitude vs TS for online DQM
  hcaldqm::ContainerSingle2D _Raddam_ADCvsEvn;  // Raddam amplitude vs Evn for local DQM
};

#endif

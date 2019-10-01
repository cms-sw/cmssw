#ifndef TestTask_h
#define TestTask_h

/*
 *	file:			TestTask.h
 *	Author:			Viktor KHristenko
 *	Description:
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"

class TestTask : public hcaldqm::DQTask {
public:
  TestTask(edm::ParameterSet const&);
  virtual ~TestTask() {}

  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&);
  virtual void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

protected:
  virtual void _process(edm::Event const&, edm::EventSetup const&);
  virtual void _resetMonitors(hcaldqm::UpdateFreq);

  //	tags
  edm::InputTag _tagHF;

  //	Hcal Filters
  hcaldqm::filter::HashFilter filter_Electronics;

  //	Electronics Map

  //	hcaldqm::Containers
  hcaldqm::Container1D _cEnergy_Subdet;
  hcaldqm::Container1D _cTiming_SubdetPMiphi;
  hcaldqm::ContainerProf1D _cEnergyvsiphi_Subdetieta;
  hcaldqm::Container2D _cEnergy_depth;
  hcaldqm::ContainerProf2D _cTiming_depth;
  hcaldqm::Container1D _cTiming_FEDSlot;
  hcaldqm::Container1D _cEnergy_CrateSpigot;
  hcaldqm::Container1D _cEnergy_FED;
  hcaldqm::Container1D _cEt_TTSubdetPM;
  hcaldqm::Container1D _cEt_TTSubdetPMiphi;
  hcaldqm::Container1D _cEt_TTSubdetieta;
  hcaldqm::Container2D _cTiming_FEDuTCA;
  hcaldqm::ContainerSingle2D _cSummary;
  hcaldqm::ContainerSingleProf1D _cPerformance;

  //		hcaldqm::ContainerProf1D	_cTiming_fCrateSlot;
  //		hcaldqm::ContainerProf1D	_cEt_TTSubdetPMiphi;
};

#endif

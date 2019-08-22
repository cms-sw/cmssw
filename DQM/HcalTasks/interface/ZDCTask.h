#ifndef ZDCTask_h
#define ZDCTask_h

/*
 *	file:			ZDCTask.h
 *	Author:			Viktor KHristenko
 *	Description:
 *		Task for ZDC Read out
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class ZDCTask : public DQMEDAnalyzer {
public:
  ZDCTask(edm::ParameterSet const&);
  ~ZDCTask() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  //	tags
  edm::InputTag _tagQIE10;
  edm::EDGetTokenT<ZDCDigiCollection> _tokQIE10;

  //	cuts/constants from input
  double _cut;
  int _ped;

  //	hcaldqm::Containers
  std::map<std::string, MonitorElement*> _cShape_EChannel;
  std::map<std::string, MonitorElement*> _cADC_EChannel;
  std::map<std::string, MonitorElement*> _cADC_vs_TS_EChannel;

  //	hcaldqm::Containers overall
  MonitorElement* _cShape;
  MonitorElement* _cADC;
  MonitorElement* _cADC_vs_TS;
};

#endif

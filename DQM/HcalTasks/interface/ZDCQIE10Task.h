#ifndef ZDCQIE10Task_h
#define ZDCQIE10Task_h

/*
 *	file:			ZDCQIE10Task.h
 *	Author:			Quan Wang
 *	Description:
 *		Task for ZDC Read out
 */

#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class ZDCQIE10Task : public DQMEDAnalyzer {
public:
  ZDCQIE10Task(edm::ParameterSet const&);
  ~ZDCQIE10Task() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  //	tags
  edm::InputTag _tagQIE10;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;

  //	hcaldqm::Containers
  std::map<uint32_t, MonitorElement*> _cADC_EChannel;
  std::map<uint32_t, MonitorElement*> _cADC_vs_TS_EChannel;
};

#endif

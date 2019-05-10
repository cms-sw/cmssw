#ifndef HcalOfflineHarvesting_h
#define HcalOfflineHarvesting_h

/**
 *	file:		HcalOffineHarvesting.h
 *	Author:		VK
 *	Date:		..
 *	Description: 
 *		This is DQMEDAnalyzer which is a edm::one module. edm::one enforces
 *		that only 1 run is being processed.
 *		https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkOneModuleInterface - for details.
 *
 *		___________
 *		Offline:
 *		For Offline only Run Summary is being generated. As it is meaningless
 *		to have current LS information being delivered. Only Total Summary
 *		makes sense
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/DQHarvester.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/DQClient.h"

#include "DQM/HcalTasks/interface/RawRunSummary.h"
#include "DQM/HcalTasks/interface/DigiRunSummary.h"
#include "DQM/HcalTasks/interface/RecoRunSummary.h"
#include "DQM/HcalTasks/interface/TPRunSummary.h"

class HcalOfflineHarvesting : public hcaldqm::DQHarvester {
public:
  HcalOfflineHarvesting(edm::ParameterSet const &);
  ~HcalOfflineHarvesting() override {}

  void beginRun(edm::Run const &, edm::EventSetup const &) override;

protected:
  void _dqmEndLuminosityBlock(DQMStore::IBooker &,
                              DQMStore::IGetter &,
                              edm::LuminosityBlock const &,
                              edm::EventSetup const &) override;
  void _dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  enum Summary { fRaw = 0, fDigi = 1, fReco = 2, fTP = 3, nSummary = 4 };

  std::vector<Summary> _summaryList;

  //	vector of Summary Generators and marks of being present
  //	by default all false
  std::map<Summary, hcaldqm::DQClient *> _sumgen;
  std::map<Summary, bool> _summarks;
  std::map<Summary, std::string> _sumnames;

  //	reportSummaryMap
  MonitorElement *_reportSummaryMap;
  MonitorElement *me;
};

#endif

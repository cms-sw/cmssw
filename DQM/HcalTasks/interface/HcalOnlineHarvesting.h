#ifndef DQM_HcalTasks_HcalOnlineHarvesting_h
#define DQM_HcalTasks_HcalOnlineHarvesting_h

/**
 *	file:		HcalOnlineHarvesting.h
 *	Author:		VK
 *	Date:		..
 *	Description: 
 *		This is DQMEDAnalyzer which is a edm::one module. edm::one enforces
 *		that only 1 run is being processed.
 *		https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkOneModuleInterface - for details.
 *
 *		HcalOnlineHarvesting is responsible for Status Evaluation and Summary
 *		Generation. In this step RAW, DIGI, RECO + TP Data Tiers 
 *		are evaluated and Summary is generated. 
 *		___________
 *		Online:
 *		There is always a Current Summary - Summary for the Current LS 
 *		being Evaluated. It might and might not include the information 
 *		from previous LSs, depending on the Quantity.
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
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/DQClient.h"

#include "DQM/HcalTasks/interface/RawRunSummary.h"
#include "DQM/HcalTasks/interface/DigiRunSummary.h"
#include "DQM/HcalTasks/interface/RecoRunSummary.h"
#include "DQM/HcalTasks/interface/TPRunSummary.h"
#include "DQM/HcalTasks/interface/PedestalRunSummary.h"

class HcalOnlineHarvesting : public hcaldqm::DQHarvester {
public:
  HcalOnlineHarvesting(edm::ParameterSet const &);
  ~HcalOnlineHarvesting() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;

protected:
  void _dqmEndLuminosityBlock(DQMStore::IBooker &,
                              DQMStore::IGetter &,
                              edm::LuminosityBlock const &,
                              edm::EventSetup const &) override;
  void _dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  enum Summary { fRaw = 0, fDigi = 1, fReco = 2, fTP = 3, fPedestal = 4, nSummary = 5 };

  //	flags to harvest...
  std::vector<bool> _vmarks;
  std::vector<hcaldqm::DQClient *> _vsumgen;
  std::vector<std::string> _vnames;

  //	thresholds
  double _thresh_bad_bad;

  //	counters
  int _nBad;
  int _nTotal;

  //	summaries
  std::vector<hcaldqm::ContainerSingle2D> _vcSummaryvsLS;

  hcaldqm::Container2D _cKnownBadChannels_depth;

  //	reportSummaryMap
  MonitorElement *_reportSummaryMap;
  MonitorElement *_runSummary;

  // Efficiencies
  hcaldqm::ContainerSingle2D _cTDCCutEfficiency_depth;
  hcaldqm::ContainerSingle1D _cTDCCutEfficiency_ieta;
};

#endif

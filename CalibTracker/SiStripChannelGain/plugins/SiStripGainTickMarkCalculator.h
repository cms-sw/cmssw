#ifndef CalibTracker_SiStripChannelGain_SiStripGainTickMarkCalculator_h
#define CalibTracker_SiStripChannelGain_SiStripGainTickMarkCalculator_h
// -*- C++ -*-
//
// Package:    SiStripApvGainCalculator
// Class:      SiStripApvGainCalculator
// 
/**\class SiStripApvGainCalculator SiStripApvGainCalculator.cc CalibTracker/SiStripChannelGain/src/SiStripApvGainCalculator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giacomo Bruno
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainTickMarkCalculator.h,v 1.1 2007/07/09 11:13:08 gbruno Exp $
//
//
//#include "DQMServices/Core/interface/MonitorUserInterface.h"
//#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>

class SiStripGainTickMarkCalculator : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainTickMarkCalculator(const edm::ParameterSet&);
  ~SiStripGainTickMarkCalculator();

private:

  void algoAnalyze(const edm::Event &, const edm::EventSetup &);

  SiStripApvGain * getNewObject();

private:

  OptoScanHistograms* histos_;
  //CommissioningHistograms* histos_;
  //  sistrip::RunType runType_;
  uint32_t runNumber_;  
  //  mutable bool first_;
  //  DaqMonitorBEInterface* bei_;

  MonitorUIRoot* mui_;
/** Output .root file. */
  std::string outputFileName_;


  bool printdebug_;

};
#endif

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
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainTickMarkCalculator.h,v 1.1 2007/06/13 14:03:35 gbruno Exp $
//
//

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
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

  CommissioningHistograms* histos_;
  sistrip::RunType runType_;
  uint32_t runNumber_;  
  //  mutable bool first_;
  DaqMonitorBEInterface* bei_;


  bool printdebug_;

};
#endif

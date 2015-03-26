// -*- C++ -*-
//
// Package:    HLXMonitor
// Class:      HLXMonitor
//
/**\class HLXMonitor HLXMonitor.cc DQM/HLXMonitor/src/HLXMonitor.cc

Description: DQM Source for HLX histograms

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Adam Hunt - Princeton University
//           email:  ahunt@princeton.edu
//         Created:  Thu Jul 19 02:29:59 EDT 2007
//
//

#ifndef _HLXMONITOR_H_
#define _HLXMONITOR_H_

// system include fileshlx_dqm_sourceclient-live.cfg
#include <iostream>
#include <string>
#include <memory>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"  // Not included in example

#include "FWCore/Framework/interface/Event.h"        // Not included in example
#include "FWCore/Framework/interface/MakerMacros.h"  // Not included in example

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

class HLXMonitor : public DQMEDAnalyzer {
 public:
  typedef HCAL_HLX::LUMI_SECTION LUMI_SECTION;
  typedef HCAL_HLX::TCPReceiver TCPReceiver;
  explicit HLXMonitor(const edm::ParameterSet&);
  ~HLXMonitor();

 private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  void connectHLXTCP();

  void SetupHists(DQMStore::IBooker&);
  void SetupEventInfo(DQMStore::IBooker&);

  void FillHistograms(const LUMI_SECTION&);
  void FillHistoHFCompare(const LUMI_SECTION&);
  void FillReportSummary();
  void FillEventInfo(const LUMI_SECTION&, const edm::Event& e);

  void ResetAll();

  void EndRun();

  double getUTCtime(timeval* a, timeval* b = NULL);

  // ----------member data ---------------------------
  TCPReceiver HLXTCP;

  MonitorElement* Set1Below[36];
  MonitorElement* Set1Between[36];
  MonitorElement* Set1Above[36];
  MonitorElement* Set2Below[36];
  MonitorElement* Set2Between[36];
  MonitorElement* Set2Above[36];
  MonitorElement* ETSum[36];

  MonitorElement* HFCompareEtSum;
  MonitorElement* HFCompareOccBelowSet1;
  MonitorElement* HFCompareOccBetweenSet1;
  MonitorElement* HFCompareOccAboveSet1;
  MonitorElement* HFCompareOccBelowSet2;
  MonitorElement* HFCompareOccBetweenSet2;
  MonitorElement* HFCompareOccAboveSet2;

  MonitorElement* AvgEtSum;
  MonitorElement* AvgOccBelowSet1;
  MonitorElement* AvgOccBetweenSet1;
  MonitorElement* AvgOccAboveSet1;
  MonitorElement* AvgOccBelowSet2;
  MonitorElement* AvgOccBetweenSet2;
  MonitorElement* AvgOccAboveSet2;

  // Luminosity Monitoring
  MonitorElement* LumiAvgEtSum;
  MonitorElement* LumiAvgOccSet1;
  MonitorElement* LumiAvgOccSet2;
  MonitorElement* LumiInstantEtSum;
  MonitorElement* LumiInstantOccSet1;
  MonitorElement* LumiInstantOccSet2;
  MonitorElement* LumiIntegratedEtSum;
  MonitorElement* LumiIntegratedOccSet1;
  MonitorElement* LumiIntegratedOccSet2;

  // Sanity Check for Occupancy
  MonitorElement* SumAllOccSet1;
  MonitorElement* SumAllOccSet2;
  MonitorElement* MissingDQMDataCheck;

  // Signal and Background Levels
  MonitorElement* MaxInstLumiBX1;
  MonitorElement* MaxInstLumiBX2;
  MonitorElement* MaxInstLumiBX3;
  MonitorElement* MaxInstLumiBX4;

  MonitorElement* MaxInstLumiBXNum1;
  MonitorElement* MaxInstLumiBXNum2;
  MonitorElement* MaxInstLumiBXNum3;
  MonitorElement* MaxInstLumiBXNum4;

  // History plots - fill once per LS
  MonitorElement* HistAvgEtSumHFP;
  MonitorElement* HistAvgEtSumHFM;
  MonitorElement* HistAvgOccBelowSet1HFP;
  MonitorElement* HistAvgOccBelowSet1HFM;
  MonitorElement* HistAvgOccBetweenSet1HFP;
  MonitorElement* HistAvgOccBetweenSet1HFM;
  MonitorElement* HistAvgOccAboveSet1HFP;
  MonitorElement* HistAvgOccAboveSet1HFM;
  MonitorElement* HistAvgOccBelowSet2HFP;
  MonitorElement* HistAvgOccBelowSet2HFM;
  MonitorElement* HistAvgOccBetweenSet2HFP;
  MonitorElement* HistAvgOccBetweenSet2HFM;
  MonitorElement* HistAvgOccAboveSet2HFP;
  MonitorElement* HistAvgOccAboveSet2HFM;
  MonitorElement* BXvsTimeAvgEtSumHFP;
  MonitorElement* BXvsTimeAvgEtSumHFM;

  MonitorElement* HistAvgLumiEtSum;
  MonitorElement* HistAvgLumiOccSet1;
  MonitorElement* HistAvgLumiOccSet2;
  MonitorElement* HistInstantLumiEtSum;
  MonitorElement* HistInstantLumiOccSet1;
  MonitorElement* HistInstantLumiOccSet2;
  MonitorElement* HistInstantLumiEtSumError;
  MonitorElement* HistInstantLumiOccSet1Error;
  MonitorElement* HistInstantLumiOccSet2Error;
  MonitorElement* HistIntegratedLumiEtSum;
  MonitorElement* HistIntegratedLumiOccSet1;
  MonitorElement* HistIntegratedLumiOccSet2;

  MonitorElement* RecentInstantLumiEtSum;
  MonitorElement* RecentInstantLumiOccSet1;
  MonitorElement* RecentInstantLumiOccSet2;
  MonitorElement* RecentIntegratedLumiEtSum;
  MonitorElement* RecentIntegratedLumiOccSet1;
  MonitorElement* RecentIntegratedLumiOccSet2;

  // EventInfo Clone
  //////////////////////////////////////////////////////////////////
  /// These MEs are filled with the info from the most recent event
  ///   by the module
  //////////////////////////////////////////////////////////////////
  MonitorElement* runId_;
  MonitorElement* runStartTimeStamp_;  /// UTC time of the run start
  MonitorElement* eventId_;
  MonitorElement* lumisecId_;
  MonitorElement* eventTimeStamp_;

  //////////////////////////////////////////////////////////////////
  /// These MEs are either static or updated upon each analyze() call
  //////////////////////////////////////////////////////////////////
  MonitorElement* nUpdates_;   /// Number of collector updates (TBD)
  MonitorElement* processId_;  /// The PID associated with this job
  MonitorElement*
      processStartTimeStamp_;  /// The UTC time of the first event processed
  MonitorElement* processTimeStamp_;  /// The UTC time of the last event
  MonitorElement* processLatency_;    /// Time elapsed since the last event
  MonitorElement* processEventRate_;  /// Avg # of events in programmable window
                                      /// (default: 5 min)
  MonitorElement* processEvents_;     ///# of event processed so far
  MonitorElement* hostName_;          /// Hostname of the local machine
  MonitorElement* processName_;       /// DQM "name" of the job (eg, Hcal or DT)
  MonitorElement* workingDir_;        /// Current working directory of the job
  MonitorElement* cmsswVer_;          /// CMSSW version run for this job
  MonitorElement* dqmPatch_;          /// DQM patch version for this job
  MonitorElement* errSummary_;  /// Subdetector-specific error summary (float)
  MonitorElement*
      errSummaryEtaPhi_;  /// Subdetector-specific etaPhi summary (float)
  MonitorElement* errSummarySegment_[10];

  // Report Summary
  MonitorElement* reportSummary_;
  MonitorElement* reportSummaryMap_;

  unsigned int numActiveTowersSet1;
  unsigned int numActiveTowersSet2;

  unsigned int counter;
  unsigned char* rData;
  short int SectionComplete;

  // Parameters
  int listenPort;
  double XMIN, XMAX;
  unsigned int NBINS;
  bool Accumulate;
  std::string OutputFilePrefix;
  std::string OutputDir;
  std::string Style;  // BX, History, Distribution
  int SavePeriod;
  unsigned int NUM_HLX;
  unsigned int NUM_BUNCHES;
  unsigned int MAX_LS;
  unsigned int AquireMode;
  unsigned int TriggerBX;
  unsigned int MinLSBeforeSave;

  std::string monitorName_;
  int prescaleEvt_;

  unsigned int reconnTime;
  std::string DistribIP1;
  std::string DistribIP2;

  unsigned int set1BelowIndex;
  unsigned int set1BetweenIndex;
  unsigned int set1AboveIndex;
  unsigned int set2BelowIndex;
  unsigned int set2BetweenIndex;
  unsigned int set2AboveIndex;

  bool ResetAtNewRun;

  std::string eventInfoFolderHLX_;
  std::string eventInfoFolder_;
  std::string subSystemName_;

  unsigned int runNumLength;
  unsigned int secNumLength;

  std::string OccXAxisTitle;
  std::string OccYAxisTitle;
  std::string EtXAxisTitle;
  std::string EtYAxisTitle;

  HCAL_HLX::LUMI_SECTION lumiSection;

  bool currentRunEnded_;
  unsigned int runNumber_;
  unsigned int expectedNibbles_;
  unsigned int totalNibbles_[36];

  unsigned int HLXHFMap[36];

  unsigned int previousSection;
  unsigned int lumiSectionCount;
  int lsBinOld;
  double sectionInstantSumEt;
  double sectionInstantErrSumEt;
  double sectionInstantSumOcc1;
  double sectionInstantErrSumOcc1;
  double sectionInstantSumOcc2;
  double sectionInstantErrSumOcc2;
  double sectionInstantNorm;

  // EventInfo Parameters
  timeval currentTime_, lastUpdateTime_, lastAvgTime_;
  timeval runStartTime_;
  float evtRateWindow_;
  int evtRateCount_;
  int pEvent_;

  // Lumi section info
  double num4NibblePerLS_;
};

#endif

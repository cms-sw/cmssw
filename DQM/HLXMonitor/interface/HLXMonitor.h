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
// $Id: HLXMonitor.h,v 1.11 2009/04/08 15:14:38 ahunt Exp $
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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h" // Not included in example

#include "FWCore/Framework/interface/Event.h" // Not included in example
#include "FWCore/Framework/interface/MakerMacros.h" // Not included in example

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoLuminosity/TCPReceiver/interface/TCPReceiver.h"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

using namespace HCAL_HLX;

//
// class decleration
//


class HLXMonitor : public edm::EDAnalyzer 
{

   public:
      explicit HLXMonitor(const edm::ParameterSet&);
      ~HLXMonitor();
      
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      //virtual void endRun(const edm::Run&, const edm::EventSetup&);

      void SaveDQMFile();

      void SetupHists();
      void SetupEventInfo();

      void FillHistograms(const LUMI_SECTION&);
      void FillHistoHFCompare(const LUMI_SECTION&);
      void FillReportSummary();
      void FillEventInfo(const LUMI_SECTION&);

      void ResetAll();

      void EndRun( bool saveFile = true );

      //  void FillHistoHistory(const LUMI_SECTION&);

      // ----------member data ---------------------------
      TCPReceiver HLXTCP;      

      MonitorElement * Set1Below[36];
      MonitorElement * Set1Between[36];
      MonitorElement * Set1Above[36];
      MonitorElement * Set2Below[36];
      MonitorElement * Set2Between[36];
      MonitorElement * Set2Above[36];
      MonitorElement * ETSum[36];

      MonitorElement * HFCompareEtSum;
      MonitorElement * HFCompareOccBelowSet1;
      MonitorElement * HFCompareOccBetweenSet1;
      MonitorElement * HFCompareOccAboveSet1;
      MonitorElement * HFCompareOccBelowSet2;
      MonitorElement * HFCompareOccBetweenSet2;
      MonitorElement * HFCompareOccAboveSet2;

      MonitorElement * AvgEtSum;
      MonitorElement * AvgOccBelowSet1;
      MonitorElement * AvgOccBetweenSet1;
      MonitorElement * AvgOccAboveSet1;
      MonitorElement * AvgOccBelowSet2;
      MonitorElement * AvgOccBetweenSet2;
      MonitorElement * AvgOccAboveSet2;

      // Luminosity Monitoring
      MonitorElement * LumiAvgEtSum;
      MonitorElement * LumiAvgOccSet1;
      MonitorElement * LumiAvgOccSet2;
      MonitorElement * LumiInstantEtSum;
      MonitorElement * LumiInstantOccSet1;
      MonitorElement * LumiInstantOccSet2;
      MonitorElement * LumiIntegratedEtSum;
      MonitorElement * LumiIntegratedOccSet1;
      MonitorElement * LumiIntegratedOccSet2;

      // Sanity Check for Occupancy
      MonitorElement * SumAllOccSet1;
      MonitorElement * SumAllOccSet2;

      // History plots - fill once per LS
      MonitorElement * HistAvgEtSumHFP;
      MonitorElement * HistAvgEtSumHFM;
      MonitorElement * HistAvgOccBelowSet1HFP;
      MonitorElement * HistAvgOccBelowSet1HFM;
      MonitorElement * HistAvgOccBetweenSet1HFP;
      MonitorElement * HistAvgOccBetweenSet1HFM;
      MonitorElement * HistAvgOccAboveSet1HFP;
      MonitorElement * HistAvgOccAboveSet1HFM;
      MonitorElement * HistAvgOccBelowSet2HFP;
      MonitorElement * HistAvgOccBelowSet2HFM;
      MonitorElement * HistAvgOccBetweenSet2HFP;
      MonitorElement * HistAvgOccBetweenSet2HFM;
      MonitorElement * HistAvgOccAboveSet2HFP;
      MonitorElement * HistAvgOccAboveSet2HFM;
      MonitorElement * BXvsTimeAvgEtSumHFP;
      MonitorElement * BXvsTimeAvgEtSumHFM;

      MonitorElement * HistAvgLumiEtSum;
      MonitorElement * HistAvgLumiOccSet1;
      MonitorElement * HistAvgLumiOccSet2;
      MonitorElement * HistInstantLumiEtSum;
      MonitorElement * HistInstantLumiOccSet1;
      MonitorElement * HistInstantLumiOccSet2;
      MonitorElement * HistIntegratedLumiEtSum;
      MonitorElement * HistIntegratedLumiOccSet1;
      MonitorElement * HistIntegratedLumiOccSet2;

      MonitorElement * RecentInstantLumiEtSum;
      MonitorElement * RecentInstantLumiOccSet1;
      MonitorElement * RecentInstantLumiOccSet2;
      MonitorElement * RecentIntegratedLumiEtSum;
      MonitorElement * RecentIntegratedLumiOccSet1;
      MonitorElement * RecentIntegratedLumiOccSet2;


      //EventInfo Clone
      //////////////////////////////////////////////////////////////////
      ///These MEs are filled with the info from the most recent event 
      ///   by the module
      //////////////////////////////////////////////////////////////////
      MonitorElement * runId_;
      MonitorElement * lumisecId_;

      // Report Summary
      MonitorElement * reportSummary_;
      MonitorElement * reportSummaryMap_;

      // DQM Store ...
      DQMStore* dbe_;

      unsigned int numActiveTowersSet1;
      unsigned int numActiveTowersSet2;

      unsigned int counter;
      unsigned char *rData;
      short int SectionComplete;
  
      // Parameters
      int listenPort;
      double XMIN,  XMAX;
      unsigned int NBINS;
      bool Accumulate;  
      std::string OutputFilePrefix;
      std::string OutputDir;
      std::string Style;   // BX, History, Distribution
      int SavePeriod;
      unsigned int NUM_HLX;
      unsigned int NUM_BUNCHES;
      unsigned int MAX_LS;
      unsigned int AquireMode;
      unsigned int TriggerBX;

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
      bool SaveAtEndJob;

      std::string eventInfoFolder_;
      std::string subSystemName_;

      unsigned int runNumLength;
      unsigned int secNumLength;

      std::string OccXAxisTitle;
      std::string OccYAxisTitle;
      std::string EtXAxisTitle;
      std::string EtYAxisTitle;

      HCAL_HLX::LUMI_SECTION lumiSection;

      unsigned int runNumber_;
      unsigned int expectedNibbles_;
      unsigned int totalNibbles_[36];

      unsigned int HLXHFMap[36];

      unsigned int lumiSectionCount;
      int lsBinOld;
      double sectionInstantSumEt;
      double sectionInstantErrSumEt;
      double sectionInstantSumOcc1;
      double sectionInstantErrSumOcc1;
      double sectionInstantSumOcc2;
      double sectionInstantErrSumOcc2;
      double sectionInstantNorm;

};

#endif

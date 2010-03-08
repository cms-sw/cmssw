// -*-c++-*-
// 
//
// $Id: HLTScalers.h,v 1.15 2010/02/11 00:11:05 wmtan Exp $
// Class to collect HLT scaler information 
// for Trigger Cross Section Monitor
// [wittich 11/07] 

// $Log: HLTScalers.h,v $
// Revision 1.15  2010/02/11 00:11:05  wmtan
// Adapt to moved framework header
//
// Revision 1.14  2010/02/02 11:42:53  wittich
// new diagnostic histograms
//
// Revision 1.13  2009/11/20 00:39:21  lorenzo
// fixes
//
// Revision 1.12  2008/09/03 13:59:05  wittich
// make HLT DQM path configurable via python parameter,
// which defaults to HLT/HLTScalers_EvF
//
// Revision 1.11  2008/09/03 02:13:47  wittich
// - bug fix in L1Scalers
// - configurable dqm directory in L1SCalers
// - other minor tweaks in HLTScalers
//
// Revision 1.10  2008/09/02 02:37:21  wittich
// - split L1 code from HLTScalers into L1Scalers
// - update cfi file accordingly
// - make sure to cd to correct directory before booking ME's
//
// Revision 1.9  2008/08/22 20:56:55  wittich
// - add client for HLT Scalers
// - Move rate calculation to HLTScalersClient and slim down the
//   filter-farm part of HLTScalers
//
// Revision 1.8  2008/08/15 15:40:57  wteo
// split hltScalers into smaller histos, calculate rates
//
// Revision 1.7  2008/08/01 14:37:33  bjbloom
// Added ability to specify which paths are cross-correlated
//

#ifndef HLTSCALERS_H
#define HLTSCALERS_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Utilities/interface/InputTag.h"

class HLTScalers: public edm::EDAnalyzer
{
public:
  /// Constructors
  HLTScalers(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~HLTScalers() {};
  
  /// BeginJob
  void beginJob(void);

//   /// Endjob
//   void endJob(void);
  
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  
//   /// Begin LumiBlock
//   void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
//                             const edm::EventSetup& c) ;

  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;


private:
  DQMStore * dbe_;
  MonitorElement *scalers_;
  MonitorElement *scalersException_;
  MonitorElement *hltCorrelations_;
  MonitorElement *detailedScalers_;
  std::string folderName_; // dqm folder name
  MonitorElement *nProc_;
  MonitorElement *nLumiBlock_;
  
  MonitorElement *hltBx_, *hltBxVsPath_;
  MonitorElement *hltOverallScaler_;
  MonitorElement *diagnostic_;

  std::vector<MonitorElement*> hltPathNames_;
  edm::InputTag trigResultsSource_;
  bool resetMe_, sentPaths_, monitorDaemon_; 

  int nev_; // Number of events processed
  int nLumi_; // number of lumi blocks
  int currentRun_;

};

#endif // HLTSCALERS_H

// -*-c++-*-
//
//
// Class to collect HLT scaler information
// for Trigger Cross Section Monitor
// [wittich 11/07]

// Revision 1.19  2011/03/29 09:46:03  rekovic
// clean vector pairPDPaths in beginRun and tidy up
//
// Revision 1.18  2011/03/24 18:25:45  rekovic
// Add single 1D plot of streamA content
//
// Revision 1.17  2010/03/17 20:54:51  wittich
// add scalers that I manually reset on beginLumi
//
// Revision 1.16  2010/02/24 17:43:47  wittich
// - keep trying to get path names if it doesn't work first time
// - move the Bx histograms out of raw to the toplevel directory.
//
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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class HLTScalers : public DQMEDAnalyzer {
 public:
  HLTScalers(const edm::ParameterSet &ps);
  virtual ~HLTScalers(){};
  void beginJob(void);
  void dqmBeginRun(const edm::Run &run, const edm::EventSetup &c);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &,
                      edm::EventSetup const &) override;
  void beginLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
                            const edm::EventSetup &c);
  void analyze(const edm::Event &e, const edm::EventSetup &c);
  /// DQM Client Diagnostic should be performed here:
  void endLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
                          const edm::EventSetup &c);
  void endRun(const edm::Run &run, const edm::EventSetup &c);

 private:
  HLTConfigProvider hltConfig_;
  std::string folderName_;  // dqm folder name
  std::string processname_;
  std::vector<std::pair<std::string, std::vector<std::string> > > pairPDPaths_;
  edm::EDGetTokenT<edm::TriggerResults> trigResultsSource_;

  MonitorElement *scalersPD_;
  MonitorElement *scalers_;
  MonitorElement *scalersN_;
  MonitorElement *scalersException_;
  MonitorElement *hltCorrelations_;
  MonitorElement *detailedScalers_;
  MonitorElement *nProc_;
  MonitorElement *nLumiBlock_;
  MonitorElement *hltBx_, *hltBxVsPath_;
  MonitorElement *hltOverallScaler_;
  MonitorElement *hltOverallScalerN_;
  MonitorElement *diagnostic_;

  bool resetMe_, sentPaths_, monitorDaemon_;

  int nev_;    // Number of events processed
  int nLumi_;  // number of lumi blocks
  int currentRun_;
};

#endif  // HLTSCALERS_H

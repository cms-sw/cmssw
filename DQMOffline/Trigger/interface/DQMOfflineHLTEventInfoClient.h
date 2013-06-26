#ifndef DQM_HLTMONITORCLIENT_DQM_H
#define DQM_HLTMONITORCLIENT_DQM_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile2D.h>

class DQMOfflineHLTEventInfoClient: public edm::EDAnalyzer {

public:

  /// Constructor
  DQMOfflineHLTEventInfoClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMOfflineHLTEventInfoClient();
 
protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  void initialize();
  edm::ParameterSet parameters_;

  DQMStore* dbe_;  
  bool verbose_;
  int counterLS_;      ///counter
  int counterEvt_;     ///counter
  int prescaleLS_;     ///units of lumi sections
  int prescaleEvt_;    ///prescale on number of events
  // -------- member data --------

  MonitorElement * reportSummary_;
  std::vector<MonitorElement*> reportSummaryContent_;
  MonitorElement * reportSummaryMap_;

  MonitorElement * CertificationSummary_;
  std::vector<MonitorElement*> CertificationSummaryContent_;
  MonitorElement * CertificationSummaryMap_;


};


#endif

#ifndef DQM_L1TMONITORCLIENT_L1TRPCTFClient_H
#define DQM_L1TMONITORCLIENT_L1TRPCTFClient_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

class L1TRPCTFClient: public edm::EDAnalyzer {

public:

  /// Constructor
  L1TRPCTFClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1TRPCTFClient();
 
protected:

  /// BeginJob
  void beginJob(void);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context);

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  void initialize();
  
  void processHistograms();


  MonitorElement * m_phipackedbad;
  MonitorElement * m_phipackeddead;
  MonitorElement * m_deadChannels;
  MonitorElement * m_noisyChannels;
  
  edm::ParameterSet parameters_;
  DQMStore* dbe_;  
  std::string monitorName_;
  std::string input_dir_;
  std::string output_dir_;
  int counterLS_;      ///counter
  int counterEvt_;     ///counter
  int prescaleLS_;     ///units of lumi sections
  int prescaleEvt_;    ///prescale on number of events

  bool verbose_;

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;



};

#endif

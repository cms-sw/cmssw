#ifndef DQM_L1TMONITORCLIENT_L1TGMTCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TGMTCLIENT_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

class L1TGMTClient: public edm::EDAnalyzer {

public:

  /// Constructor
  L1TGMTClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1TGMTClient();
 
protected:

  /// BeginJob
  void beginJob(void);

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
  void process();
  void makeRatio1D(MonitorElement* mer, std::string h1Name, std::string h2Name);
  void makeEfficiency1D(MonitorElement *meeff, std::string heName, std::string hiName); 
  void makeEfficiency2D(MonitorElement *meeff, std::string heName, std::string hiName); 
  TH1F * get1DHisto(std::string meName, DQMStore* dbi);
  TH2F * get2DHisto(std::string meName, DQMStore* dbi);
  MonitorElement* bookClone1D(std::string name, std::string title, std::string hrefName);
  MonitorElement* bookClone1DVB(std::string name, std::string title, std::string hrefName);
  MonitorElement* bookClone2D(std::string name, std::string title, std::string hrefName);
  
  edm::ParameterSet parameters_;
  DQMStore* dbe_;  
  std::string monitorName_;
  std::string input_dir_;
  std::string output_dir_;
  int counterLS_;      ///counter
  int counterEvt_;     ///counter
  int prescaleLS_;     ///units of lumi sections
  int prescaleEvt_;    ///prescale on number of events

  // -------- member data --------
  MonitorElement* eff_eta_dtcsc;
  MonitorElement* eff_eta_rpc;
  MonitorElement* eff_phi_dtcsc;
  MonitorElement* eff_phi_rpc;
  MonitorElement* eff_etaphi_dtcsc;
  MonitorElement* eff_etaphi_rpc;
  
};

#endif

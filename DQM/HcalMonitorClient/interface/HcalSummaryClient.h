#ifndef HcalSummaryClient_H
#define HcalSummaryClient_H

/*
 * \file HcalSummaryClient.h
 *
 * Code ported from DQM/EcalBarrelMonitorClient/interface/EBSummaryClient.h
 * $Date: 2008/04/08 18:04:48 $
 * $Revision: 1.35 $
 * \author Jeff Temple
 *
*/

#include <vector>
#include <string>
#include <fstream>

#include "TROOT.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"

class MonitorElement;
class DQMStore;

class HcalSummaryClient : public HcalBaseClient {

 public:

  // Constructor
   
  HcalSummaryClient(const edm::ParameterSet& ps);
   
  // Destructor
  virtual ~HcalSummaryClient();
     
  // BeginJob
  void beginJob(DQMStore* dqmStore);
    
  // EndJob
  void endJob(void);
  
  // BeginRun
  void beginRun(void);
  
  // EndRun
  void endRun(void);
  
  // Setup
  void setup(void);
  
  // Cleanup
  void cleanup(void);
  

  // Analyze
  void analyze(void);

  // HtmlOutput
  void htmlOutput(int run, std::string& htmlDir, std::string& htmlName);
  
  // Get Functions
  inline int getEvtPerJob() { return ievt_; }
  inline int getEvtPerRun() { return jevt_; }


 private:

  int ievt_;
  int jevt_;
  
  bool cloneME_;
  
  bool verbose_;
  bool debug_;
  
  std::string prefixME_;
  
  bool enableCleanup_;

  DQMStore* dqmStore_;

  MonitorElement* meGlobalSummary_;
  MonitorElement* meHotCellMap_;
  MonitorElement* meDeadCellMap_;

  double etaMin_, etaMax_, phiMin_, phiMax_;
  int phiBins_, etaBins_;
    
}; // end of class declaration

#endif

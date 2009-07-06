#ifndef HcalSummaryClient_H
#define HcalSummaryClient_H

/*
 * \file HcalSummaryClient.h
 *
 * Code ported from DQM/EcalBarrelMonitorClient/interface/EBSummaryClient.h
 * $Date: 2009/06/28 20:24:09 $
 * $Revision: 1.15.2.5 $
 * \author Jeff Temple
 *
*/

#include <vector>
#include <string>
#include <fstream>
#include <sys/time.h>

#include <memory>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <ostream>

#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"

#include "DQM/HcalMonitorClient/interface/HcalDataFormatClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDigiClient.h"
#include "DQM/HcalMonitorClient/interface/HcalRecHitClient.h"
#include "DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h"
#include "DQM/HcalMonitorClient/interface/HcalPedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDeadCellClient.h"
#include "DQM/HcalMonitorClient/interface/HcalHotCellClient.h"
#include "DQM/HcalMonitorClient/interface/SubTaskSummaryStatus.h"

class MonitorElement;
class DQMStore;

class HcalSummaryClient : public HcalBaseClient {

 public:

  // Constructor
   
  HcalSummaryClient();
   
  // Destructor
  virtual ~HcalSummaryClient();
     
  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

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
  void analyze_subtask(SubTaskSummaryStatus& s);
  void New_analyze_subtask(SubTaskSummaryStatus& s);
  void resetSummaryPlot(int Subdet);
  void incrementCounters(void);

  // HtmlOutput
  void htmlOutput(int& run, time_t& mytime, int& minlumi, int& maxlumi, std::string& htmlDir, std::string& htmlName);
  
  // Get Functions
  inline int getEvtPerJob() { return ievt_; }
  inline int getEvtPerRun() { return jevt_; }

 // Introduce temporary error/warning checks
  bool hasErrors_Temp();
  bool hasWarnings_Temp();
  bool hasOther_Temp() {return false;}

 private:

  int ievt_;
  int jevt_;
  int lastupdate_;

  int HBpresent_, HEpresent_, HOpresent_, HFpresent_, ZDCpresent_;

  bool cloneME_;

  int debug_;
  
  std::string prefixME_;
  
  bool enableCleanup_;

  DQMStore* dqmStore_;

  SubTaskSummaryStatus dataFormatMon_, digiMon_, recHitMon_;
  SubTaskSummaryStatus pedestalMon_, ledMon_, hotCellMon_;
  SubTaskSummaryStatus deadCellMon_, trigPrimMon_, caloTowerMon_;
  SubTaskSummaryStatus beamMon_;  // still needs to be added into src code

  std::map<std::string, int> subdetCells_;
  int totalcells_; // stores total possible # of cells being checked

  // overall values for each subdetector and global status
  double status_HB_;
  double status_HE_;
  double status_HO_;
  double status_HF_;
  double status_ZDC_;
  double status_global_;
    
  ofstream htmlFile;

  double etaMin_, etaMax_, phiMin_, phiMax_;
  int etaBins_, phiBins_;
}; // end of class declaration

#endif

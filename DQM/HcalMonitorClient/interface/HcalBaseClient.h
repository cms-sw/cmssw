#ifndef HcalBaseClient_H
#define HcalBaseClient_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/CPUTimer.h" 

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMOldReceiver.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TColor.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>



// Don't like having these here in the header; can we move them to src?
using namespace cms;
using namespace edm;
using namespace std;

class HcalBaseClient{
  
 public:
  
  /// Constructor
  HcalBaseClient();
  
  /// Destructor
  virtual ~HcalBaseClient();
  
  virtual void init(const ParameterSet& ps, DQMStore* dbe_, string clientName);

  void errorOutput();
  void getTestResults(int& totalTests, 
		      map<string, vector<QReport*> >& err, 
		      map<string, vector<QReport*> >& warn, 
		      map<string, vector<QReport*> >& other);
  bool hasErrors() const { return dqmReportMapErr_.size(); }
  bool hasWarnings() const { return dqmReportMapWarn_.size(); }
  bool hasOther() const { return dqmReportMapOther_.size(); }

  // Introduce temporary error/warning checks
  bool hasErrors_Temp()  {return false;}
  bool hasWarnings_Temp()  {return false;}
  bool hasOther_Temp()  {return false;}

  bool validDetId(HcalSubdetector sd, int ies, int ip, int dp);
  
  void getSJ6histos( char* dir, char* name, TH2F* h[6]);
  void getSJ6histos( char* dir, char* name, TH1F* h[6]);

 protected:

  int ievt_;
  int jevt_;
  
  bool cloneME_;
  int debug_;
  string process_;
  string baseFolder_;
  string clientName_;
  
  bool showTiming_; // controls whether to show timing diagnostic info 
  edm::CPUTimer cpu_timer; //  
  bool fillUnphysical_; // determine whether or not to fill unphysical cells in iphi

  DQMStore* dbe_;
  
  bool subDetsOn_[4];
  
  // Define standard error palette
  int pcol_error_[20];
  float rgb_error_[20][3];


  // Quality criteria for data integrity
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;

};

#endif

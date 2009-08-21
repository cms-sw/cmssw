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
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Don't like having these here in the header; can we move them to src?
using namespace cms;
using namespace edm;
using namespace std;

// Structs that will be the actual data mapped
// Contain fileds for the array of histograms, the name of the histos,
// and the file path within the ROOT file.
typedef struct {
  TH1F **hist;
  string name;
  string file;
} Hist1DContext;
typedef struct {
  TH2F **hist;
  string name;
  string file;
} Hist2DContext;
typedef struct {
  TProfile **hist;
  string name;
  string file;
} HistProfileContext;


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
  
  void getSJ6histos( std::string dir, std::string name, TH2F* h[6], std::string units="");
  void getSJ6histos( std::string dir, std::string name, TH1F* h[4], std::string units="");

  bool vetoCell(HcalDetId& id);

  void getEtaPhiHists( std::string dir, std::string name, TH2F* h[4], std::string units=""); // assumes base directory of "Hcal"
  void getEtaPhiHists(std::string rootdir, std::string dir, std::string name, TH2F* h[4], std::string units="");
  void SetEtaPhiLabels(MonitorElement* x);
  
  /*
    int CalcIeta(int hist_eta, int depth);
    bool isHB(int etabin, int depth);
    bool isHE(int etabin, int depth);
    bool isHO(int etabin, int depth);
    bool isHF(int etabin, int depth);
  */

 protected:

  int ievt_;
  int jevt_;
  
  bool cloneME_;
  int debug_;
  std::string process_;
  std::string rootFolder_;
  std::string clientName_;
  
  bool showTiming_; // controls whether to show timing diagnostic info 
  edm::CPUTimer cpu_timer; //  
  bool fillUnphysical_; // determine whether or not to fill unphysical cells in iphi

  DQMStore* dbe_;
  
  bool subDetsOn_[4];
  
  // Define standard error palette
  int pcol_error_[20];
  float rgb_error_[20][3];

  static const int binmapd2[];
  static const int binmapd3[];

  std::vector <std::string> badCells_;

  // Quality criteria for data integrity
  map<string, vector<QReport*> > dqmReportMapErr_;
  map<string, vector<QReport*> > dqmReportMapWarn_;
  map<string, vector<QReport*> > dqmReportMapOther_;
  map<string, string> dqmQtests_;

  // Map type: use HistMap?D_t::iterator as the type for your iterator
  typedef map<string, Hist1DContext> HistMap1D_t;
  typedef map<string, Hist2DContext> HistMap2D_t;
  typedef map<string, HistProfileContext> HistMapProfile_t;

  HistMap1D_t histMap1D;
  HistMap2D_t histMap2D;
  HistMapProfile_t histMapProfile;

  void mapHist1D(string, string, TH1F*[]);
  void mapHist2D(string, string, TH2F*[]);
  void mapHistProfile(string, string, TProfile*[]);

  // Monitor Elements for calculating error rates
  MonitorElement* ProblemCells;
  EtaPhiHists ProblemCellsByDepth;

};

#endif

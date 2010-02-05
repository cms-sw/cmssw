#ifndef HcalMonitorClient_H
#define HcalMonitorClient_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/CPUTimer.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
          
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalSummaryClient.h>
#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>

// ############################################################
#include <DQM/HcalMonitorClient/interface/HcalDetDiagNoiseMonitorClient.h>
// ############################################################

#include <DQM/HcalMonitorClient/interface/HcalPedestalClient.h>
#include <DQM/HcalMonitorClient/interface/HcalLEDClient.h>
#include <DQM/HcalMonitorClient/interface/HcalLaserClient.h>
#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalDeadCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalTrigPrimClient.h>
#include <DQM/HcalMonitorClient/interface/HcalCaloTowerClient.h>
#include <DQM/HcalMonitorClient/interface/HcalBeamClient.h>

/////////////////////////////////////////////////////////////////
#include "DQM/HcalMonitorClient/interface/HcalDetDiagPedestalClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagLEDClient.h"
#include "DQM/HcalMonitorClient/interface/HcalDetDiagLaserClient.h"
/////////////////////////////////////////////////////////////////

#include <DQM/HcalMonitorClient/interface/HcalDQMDbInterface.h>

// Use to hold/get channel status
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "DataFormats/DetId/interface/DetId.h"


#include "TROOT.h"
#include "TTree.h"
#include "TGaxis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

using namespace std;

class HcalMonitorClient : public EDAnalyzer{
  
public:
  
  /// Constructors
  HcalMonitorClient();
  HcalMonitorClient(const ParameterSet& ps);
  
  /// Destructor
  ~HcalMonitorClient();
  
  // Initialize
  void initialize(const ParameterSet& ps);
  void offlineSetup();

  /// Analyze
  void analyze(void);
  void analyze(const Event& evt, const EventSetup& es);
  
  /// BeginJob
  void beginJob();
  /// BeginRun
  void beginRun(const Run& r, const edm::EventSetup & c);
  /// BeginLumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

  /// EndJob
  void endJob(void);
  /// EndRun
  void endRun(const edm::Run & r, const edm::EventSetup & c);
  /// EndLumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);
  
  /// HtmlOutput
  void htmlOutput(void);

  /// Create reports
  void report(bool update);

  /// Generate error summary
  void errorSummary();

  /// Create tests
  void createTests(void);

  /// reset all monitor elements
  void resetAllME(void);

  //Offline output functions
  void loadHistograms(TFile* infile, const char* fname);
  void dumpHistograms(int& runNum, vector<TH1F*> &hist1d, vector<TH2F*> &hist2d);

  /// Boolean prescale test for this event
  bool prescale();

 private:
  void removeAllME(void);
  void writeDBfile();
  /********************************************************/
  //  The following member variables can be specified in  //
  //  the configuration input file for the process.       //
  /********************************************************/

  /// Prescale variables for restricting the frequency of analyzer
  /// behavior.  The base class does not implement prescales.
  /// Set to -1 to be ignored.
  int prescaleEvt_;    ///units of events
  int prescaleLS_;     ///units of lumi sections
  int prescaleTime_;   ///units of minutes
  int prescaleUpdate_; ///units of "updates", TBD

  /// The name of the monitoring process which derives from this
  /// class, used to standardize filename and file structure
  std::string monitorName_;

  /// Verbosity switch used for debugging or informational output
  int debug_ ;
  
  // Timing diagnostic switch
  bool showTiming_; // controls whether to show timing diagnostic info 
  edm::CPUTimer cpu_timer; //  

  /// counters and flags
    //int nevt_; // counts number of events actually analyzed by HcalMonitorClient
  int nlumisecs_;
  bool saved_;
  bool Online_;
  
  struct{
    timeval startTV,updateTV;
    double startTime;
    double elapsedTime; 
    double updateTime;
  } psTime_;    
  
  ///Connection to the DQM backend
  DQMStore* dbe_;  
  
  // environment variables
  int irun_,ievent_,itime_;
  int ilumisec_;
  int maxlumisec_, minlumisec_;

  time_t mytime_;

  std::string rootFolder_;

  int ievt_; // counts number of events read by client (and analyzed by tasks)
  int resetEvents_;
  int resetLS_;
  
  bool runningStandalone_;
  bool enableMonitorDaemon_;

  string inputFile_;
  string baseHtmlDir_;

  HcalSummaryClient*         summary_client_;
  HcalDataFormatClient*      dataformat_client_;
  HcalDigiClient*            digi_client_;
  HcalRecHitClient*          rechit_client_;

// ############################################################
  HcalDetDiagNoiseMonitorClient*          noise_client_;
// ############################################################

  HcalPedestalClient*        pedestal_client_;
  HcalLEDClient*             led_client_;
  HcalLaserClient*           laser_client_;
  HcalHotCellClient*         hot_client_;
  HcalDeadCellClient*        dead_client_;
  HcalTrigPrimClient*        tp_client_;
  HcalCaloTowerClient*       ct_client_;
  HcalBeamClient*            beam_client_;

  ///////////////////////////////////////////////////////////
  HcalDetDiagPedestalClient* detdiagped_client_; 
  HcalDetDiagLEDClient*      detdiagled_client_; 
  HcalDetDiagLaserClient*    detdiaglas_client_; 
  //////////////////////////////////////////////////

  // myquality_ will store status values for each det ID I find
  std::string databasedir_; // empty string means don't dump out db info
  std::map<HcalDetId, unsigned int> myquality_;
  HcalChannelQuality* chanquality_;

};

#endif

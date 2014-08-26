#ifndef ZDCMonitorClient_H
#define ZDCMonitorClient_H


#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/CPUTimer.h"

class TH2F;
class TH1F;
class TFile;

class ZDCMonitorClient : public DQMEDAnalyzer{
  
public:
  
  /// Constructors
  ZDCMonitorClient();
  ZDCMonitorClient(const edm::ParameterSet& ps);
  
  /// Destructor
  ~ZDCMonitorClient();
  
  // Initialize
  void initialize(const edm::ParameterSet& ps);
  void offlineSetup();

  // book histograms
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const & );

  /// Analyze
  void analyze(void);
  void analyze(const edm::Event& evt, const edm::EventSetup& es);
  
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
  void dumpHistograms(int& runNum, std::vector<TH1F*> &hist1d, std::vector<TH2F*> &hist2d);

  /// Boolean prescale test for this event
  bool prescale();

 private:
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
    //int nevt_; // counts number of events actually analyzed by ZDCMonitorClient
  int nlumisecs_;
  bool saved_;
  bool Online_;
  
  struct{
    timeval startTV,updateTV;
    double startTime;
    double elapsedTime; 
    double updateTime;
  } psTime_;    
  
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

  std::vector<MonitorElement *> mes_;
  MonitorElement * errorSummary_;

  std::string inputFile_;
  std::string baseHtmlDir_;

};

#endif

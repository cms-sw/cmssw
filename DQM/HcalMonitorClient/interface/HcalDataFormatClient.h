#ifndef HcalDataFormatClient_H
#define HcalDataFormatClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HcalDataFormatClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalDataFormatClient();
  
  /// Destructor
  ~HcalDataFormatClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);    

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob(void);
  
  /// EndJob
  void endJob(void);
  
  /// BeginRun
  void beginRun(void);
  
  /// EndRun
  void endRun(void);
  
  /// Setup
  void setup(void);
  
  /// Cleanup
  void cleanup(void);
  
  void report();
  
  /// HtmlOutput
  void htmlOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  void resetAllME();
  void createTests();

 private:
  
  void labelxBits(TH1F* hist);
  void labelyBits(TH2F* hist);
  
  TH1F* spigotErrs_;
  TH1F* badDigis_;
  TH1F* unmappedDigis_;
  TH1F* unmappedTPDs_;
  TH1F* fedErrMap_;
  TH1F* BCN_;
  TH1F* dccBCN_;
  
  TH1F* BCNCheck_;
  TH1F* EvtNCheck_;
  TH1F* FibOrbMsgBCN_;
  
  TH1F* dferr_[3];

  TH2F* DCC_Err_Warn_;
  TH2F* CDF_Violation_;
  TH2F* DCC_Evt_Fmt_;
  TH2F* DCC_Spigot_Err_;
  TH2F* BCNMap_;
  TH2F* EvtMap_;
  TH2F* ErrMapbyCrate_;
  TH2F* FWVerbyCrate_;
  TH2F* ErrCrate0_;
  TH2F* ErrCrate1_;
  TH2F* ErrCrate2_;
  TH2F* ErrCrate3_;
  TH2F* ErrCrate4_;
  TH2F* ErrCrate5_;
  TH2F* ErrCrate6_;
  TH2F* ErrCrate7_;
  TH2F* ErrCrate8_;
  TH2F* ErrCrate9_;
  TH2F* ErrCrate10_;
  TH2F* ErrCrate11_;
  TH2F* ErrCrate12_;
  TH2F* ErrCrate13_;
  TH2F* ErrCrate14_;
  TH2F* ErrCrate15_;
  TH2F* ErrCrate16_;
  TH2F* ErrCrate17_;

  TH2F* InvHTRData_;

};

#endif

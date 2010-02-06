#ifndef HcalDataFormatClient_H
#define HcalDataFormatClient_H
#define NUMDCCS 32
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
  void getHistograms(bool getEmAll=false);
  void loadHistograms(TFile* f);
  
  void resetAllME();
  void createTests();

 private:
  
  void labelxBits(TH1F* hist);
  void labelyBits(TH2F* hist);
 
  //In the rough order that the DFMonitor fills these:
  TH2F*  CDF_Violation_;      //Summary histo of Common Data Format violations by FED ID
  TH2F*  DCCEventFormatError_;//Summary histo of DCC Event Format violations by FED ID 
  TH2F*  DCCStatusBits_;  
  TProfile* DCCVersion_;
  TH1F*  FEDRawDataSizes_;
  TProfile*  EvFragSize_;
  TH2F*  EvFragSize2_;
  TH1F*  FEDEntries_;

  TH2F*  LRBDataCorruptionIndicators_; 

  TH1F*  HTRBCN_;         // Bunch count number distributions
  TH1F*  dccBCN_;         // Bunch count number distributions
  TH1F*  BCNCheck_;       // HTR BCN compared to DCC BCN
  TH2F*  BCNSynch_;       // htr-htr disagreement location
  TH1F*  EvtNCheck_;      // HTR Evt # compared to DCC Evt #
  TH2F*  EvtNumberSynch_; // htr-htr disagreement location
  TH1F*  OrNCheck_;       // htr OrN compared to dcc OrN
  TH2F*  OrNSynch_;       // htr-htr disagreement location
  TH1F*  BCNwhenOrNDiff_; // BCN distribution (subset)

  TH2F*  HalfHTRDataCorruptionIndicators_;
  TH2F*  DataFlowInd_;
  TH2F*  InvHTRData_;
  TH2F*  HTRFWVersionByCrate_; 
  TH1F*  meUSFractSpigs_;
  TH2F*  HTRStatusWdByCrate_; //  TH2F* ErrMapbyCrate_; //HTR error bits by crate
  TH2F*  HTRStatusCrate0_;   //Map of HTR errors into Crate 0
  TH2F*  HTRStatusCrate1_;   //Map of HTR errors into Crate 1
  TH2F*  HTRStatusCrate2_;   //Map of HTR errors into Crate 2
  TH2F*  HTRStatusCrate3_;   //Map of HTR errors into Crate 3
  TH2F*  HTRStatusCrate4_;   //Map of HTR errors into Crate 4
  TH2F*  HTRStatusCrate5_;   //Map of HTR errors into Crate 5
  TH2F*  HTRStatusCrate6_;   //Map of HTR errors into Crate 6
  TH2F*  HTRStatusCrate7_;   //Map of HTR errors into Crate 7
  TH2F*  HTRStatusCrate9_;   //Map of HTR errors into Crate 9
  TH2F*  HTRStatusCrate10_;  //Map of HTR errors into Crate 10
  TH2F*  HTRStatusCrate11_;  //Map of HTR errors into Crate 11
  TH2F*  HTRStatusCrate12_;  //Map of HTR errors into Crate 12
  TH2F*  HTRStatusCrate13_;  //Map of HTR errors into Crate 13
  TH2F*  HTRStatusCrate14_;  //Map of HTR errors into Crate 14
  TH2F*  HTRStatusCrate15_;  //Map of HTR errors into Crate 15
  TH2F*  HTRStatusCrate17_;  //Map of HTR errors into Crate 17
  TH1F*  HTRStatusWdByPartition_[3];

  TH2F*  ChannSumm_DataIntegrityCheck_;
  // handy array of pointers to pointers...
  TH2F* Chann_DataIntegrityCheck_[NUMDCCS];

  TH1F*  FibBCN_;
  TH2F*  Fib1OrbMsgBCN_;  //BCN of Fiber 1 Orb Msg
  TH2F*  Fib2OrbMsgBCN_;  //BCN of Fiber 2 Orb Msg
  TH2F*  Fib3OrbMsgBCN_;  //BCN of Fiber 3 Orb Msg
  TH2F*  Fib4OrbMsgBCN_;  //BCN of Fiber 4 Orb Msg
  TH2F*  Fib5OrbMsgBCN_;  //BCN of Fiber 5 Orb Msg
  TH2F*  Fib6OrbMsgBCN_;  //BCN of Fiber 6 Orb Msg
  TH2F*  Fib7OrbMsgBCN_;  //BCN of Fiber 7 Orb Msg
  TH2F*  Fib8OrbMsgBCN_;  //BCN of Fiber 8 Orb Msg
};

#endif

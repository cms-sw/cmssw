#ifndef DQM_L1TMONITORCLIENT_L1TEventInfoClient_H
#define DQM_L1TMONITORCLIENT_L1TEventInfoClient_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
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

class L1TEventInfoClient: public edm::EDAnalyzer {

public:

  /// Constructor
  L1TEventInfoClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1TEventInfoClient();
 
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
  TH1F * get1DHisto(std::string meName, DQMStore * dbi);
  TH2F * get2DHisto(std::string meName, DQMStore * dbi);
  TProfile2D * get2DProfile(std::string meName, DQMStore * dbi);
  TProfile * get1DProfile(std::string meName, DQMStore * dbi);
  edm::ParameterSet parameters_;
  std::string StringToUpper(std::string strToConvert);

  DQMStore* dbe_;  
  std::string monitorDir_;
  bool verbose_;
  int counterLS_;      ///counter
  int counterEvt_;     ///counter
  int prescaleLS_;     ///units of lumi sections
  int thresholdLS_;    ///units of lumi sections
  int prescaleEvt_;    ///prescale on number of events
  int nChannels;

  double GCT_NonIsoEm_threshold_;
  double GCT_IsoEm_threshold_;
  double GCT_TauJets_threshold_;
  double GCT_AllJets_threshold_;
  double GMT_Muons_threshold_;

  enum DataValue { data_empty, data_all, data_gt, data_muons, 
		   data_jets, data_taujets, data_isoem, 
		   data_nonisoem, data_met };
  enum EmulValue { emul_empty, emul_all, emul_gt, emul_dtf, 
		   emul_dtp, emul_ctf, emul_ctp, emul_rpc, 
		   emul_gmt, emul_etp, emul_htp, emul_rct, 
		   emul_gct, emul_glt };

  std::map<std::string, DataValue> s_mapDataValues;
  std::map<std::string, EmulValue> s_mapEmulValues;

  static const int nsys_=18;

  Float_t reportSummary;
  Float_t summarySum;
  Float_t summaryContent[20];
  std::vector<std::string> dataMask;
  std::vector<std::string> emulMask;

  // -------- member data --------

  MonitorElement * reportSummary_;
  MonitorElement * reportSummaryContent_[20];
  MonitorElement * reportSummaryMap_;


};

#endif

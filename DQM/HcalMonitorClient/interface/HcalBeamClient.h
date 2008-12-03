#ifndef GUARD_HcalBeamClient_H
#define GUARD_HcalBeamClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalBeamClient : public HcalBaseClient {

 public:

  /// Constructor
  HcalBeamClient();
  /// Destructor
  ~HcalBeamClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

 /// Analyze
  void analyze(void);

  /// BeginJob
  //void beginJob(const EventSetup& c);
  void beginJob();

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

  /// HtmlOutput
  void htmlOutput(int run, string htmlDir, string htmlName);
  void htmlExpertOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);

  ///process report
  void report();

  void resetAllME();
  void createTests();

private:

  vector <std::string> subdets_;

  double minErrorFlag_;  // minimum error rate which causes problem cells to be dumped in client
  bool beamclient_makeDiagnostics_;

  int beamclient_checkNevents_;

  // Histograms
  TH2F* ProblemBeamCells;
  TH2F* ProblemBeamCellsByDepth[6];
  TH1F* HB_CenterOfEnergyRadius[83];
  TH1F* HE_CenterOfEnergyRadius[83];
  TH1F* HF_CenterOfEnergyRadius[83];
  TH1F* HO_CenterOfEnergyRadius[83];
  TH1F* CenterOfEnergyRadius;
  TH2F* CenterOfEnergy;
  TProfile* COEradiusVSeta;

  TH1F* HBCenterOfEnergyRadius;
  TH2F* HBCenterOfEnergy;
  TH1F* HECenterOfEnergyRadius;
  TH2F* HECenterOfEnergy;
  TH1F* HOCenterOfEnergyRadius;
  TH2F* HOCenterOfEnergy;
  TH1F* HFCenterOfEnergyRadius;
  TH2F* HFCenterOfEnergy;

  TProfile* Etsum_eta_L;
  TProfile* Etsum_eta_S;
  TProfile* Etsum_phi_L;
  TProfile* Etsum_phi_S;
  TH1F* Etsum_ratio_p;
  TH1F* Etsum_ratio_m;
  TH2F* Etsum_map_L;
  TH2F* Etsum_map_S;
  TH2F* Etsum_ratio_map;
  TH2F* Etsum_rphi_L;
  TH2F* Etsum_rphi_S;
  TH1F* Energy_Occ;

  TH2F* Occ_rphi_L;
  TH2F* Occ_rphi_S;
  TProfile* Occ_eta_L;
  TProfile* Occ_eta_S;
  TProfile* Occ_phi_L;
  TProfile* Occ_phi_S;
  TH2F* Occ_map_L;
  TH2F* Occ_map_S;
  
  TH1F* HFlumi_ETsum_perwedge;
  TH1F* HFlumi_Occupancy_above_thr_r1;
  TH1F* HFlumi_Occupancy_between_thrs_r1;
  TH1F* HFlumi_Occupancy_below_thr_r1;
  TH1F* HFlumi_Occupancy_above_thr_r2;
  TH1F* HFlumi_Occupancy_between_thrs_r2;
  TH1F* HFlumi_Occupancy_below_thr_r2;

}; // class HcalBeamClient

#endif

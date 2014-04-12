#ifndef HcalRecHitClient_GUARD_H
#define HcalRecHitClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class EtaPhiHists; // forward declaration

class HcalRecHitClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalRecHitClient(){name_="";};
  HcalRecHitClient(std::string myname);//{ name_=myname;};
  HcalRecHitClient(std::string myname, const edm::ParameterSet& ps);

  void analyze(void);
  void calculateProblems(void); // calculates problem histogram contents
  void updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual);
  void beginJob(void);
  void endJob(void);
  void beginRun(void);
  void endRun(void); 
  void setup(void);  
  void cleanup(void);

  bool hasErrors_Temp(void);  
  bool hasWarnings_Temp(void);
  bool hasOther_Temp(void);
  bool test_enabled(void);
  
  /// Destructor
  ~HcalRecHitClient();

 private:
  int nevts_;

  EtaPhiHists* meEnergyByDepth;
  EtaPhiHists* meEnergyThreshByDepth;
  EtaPhiHists* meTimeByDepth;
  EtaPhiHists* meTimeThreshByDepth;
  EtaPhiHists* meSqrtSumEnergy2ByDepth;
  EtaPhiHists* meSqrtSumEnergy2ThreshByDepth;

  MonitorElement* meHBEnergy_1D;
  MonitorElement* meHEEnergy_1D;
  MonitorElement* meHOEnergy_1D;
  MonitorElement* meHFEnergy_1D;

  MonitorElement* meHBEnergyRMS_1D;
  MonitorElement* meHEEnergyRMS_1D;
  MonitorElement* meHOEnergyRMS_1D;
  MonitorElement* meHFEnergyRMS_1D;

  MonitorElement* meHBEnergyThresh_1D;
  MonitorElement* meHEEnergyThresh_1D;
  MonitorElement* meHOEnergyThresh_1D;
  MonitorElement* meHFEnergyThresh_1D;

  MonitorElement* meHBEnergyRMSThresh_1D;
  MonitorElement* meHEEnergyRMSThresh_1D;
  MonitorElement* meHOEnergyRMSThresh_1D;
  MonitorElement* meHFEnergyRMSThresh_1D;
};

#endif

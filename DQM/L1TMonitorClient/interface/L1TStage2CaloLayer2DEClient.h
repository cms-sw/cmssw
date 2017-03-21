#ifndef DQM_L1TMONITORCLIENT_L1TStage2CaloLayer2DECLIENT_H
#define DQM_L1TMONITORCLIENT_L1TStage2CaloLayer2DECLIENT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class L1TStage2CaloLayer2DEClient: public DQMEDHarvester {
  
 public:
  
  L1TStage2CaloLayer2DEClient(const edm::ParameterSet&);
  
  virtual ~L1TStage2CaloLayer2DEClient();
  
 protected:
  
  virtual void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter)override;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,DQMStore::IGetter &igetter,const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;
  
 private:
  
  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &igetter);
  
  std::string monitor_dir_;
  std::string input_dir_data_;
  std::string input_dir_emul_;
  
  MonitorElement* CenJetRankComp_;
  MonitorElement* CenJetEtaComp_;
  MonitorElement* CenJetPhiComp_;
  MonitorElement* ForJetRankComp_;
  MonitorElement* ForJetEtaComp_;
  MonitorElement* ForJetPhiComp_;
  MonitorElement* IsoEGRankComp_;
  MonitorElement* IsoEGEtaComp_;
  MonitorElement* IsoEGPhiComp_;
  MonitorElement* NonIsoEGRankComp_;
  MonitorElement* NonIsoEGEtaComp_;
  MonitorElement* NonIsoEGPhiComp_;
  MonitorElement* IsoTauRankComp_;
  MonitorElement* IsoTauEtaComp_;
  MonitorElement* IsoTauPhiComp_;
  MonitorElement* TauRankComp_;
  MonitorElement* TauEtaComp_;
  MonitorElement* TauPhiComp_;
  MonitorElement* METComp_;
  MonitorElement* MHTComp_;
  MonitorElement* ETTComp_;
  MonitorElement* HTTComp_;
};

#endif

    
  

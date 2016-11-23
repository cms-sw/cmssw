#ifndef DQM_L1TMONITORCLIENT_L1TStage2EMTFDECLIENT_H
#define DQM_L1TMONITORCLIENT_L1TStage2EMTFDECLIENT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class L1TStage2EMTFDEClient: public DQMEDHarvester {
  
 public:
  
  L1TStage2EMTFDEClient(const edm::ParameterSet&);
  
  virtual ~L1TStage2EMTFDEClient();
  
 protected:
  
  virtual void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter)override;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,DQMStore::IGetter &igetter,const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
 private:
  
  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &igetter); 
  virtual void fixZero(TH1F* histData, TH1F* histEmul);
  
  std::string monitor_dir_;
  std::string input_dir_data_;
  std::string input_dir_emul_;
  
  MonitorElement* emtfMuonhwEtaComp_; 
  MonitorElement* emtfMuonhwEtaCompCoarse_;
  MonitorElement* emtfMuonhwPhiComp_;
  MonitorElement* emtfMuonhwPhiCompCoarse_;
  MonitorElement* emtfMuonhwPtComp_;
  MonitorElement* emtfMuonhwPtCompCoarse_;
  MonitorElement* emtfMuonhwQualComp_;
  MonitorElement* emtfMuonhwBXComp_;


  MonitorElement* emtfTrackEtaComp_;
  
  MonitorElement* emtfTrackEtaCompCoarse_;
  MonitorElement* emtfTrackPhiComp_;
  MonitorElement* emtfTrackPhiCompCoarse_;
  MonitorElement* emtfTrackPtComp_;
  MonitorElement* emtfTrackPtCompCoarse_;
  MonitorElement* emtfTrackQualComp_;
  MonitorElement* emtfTrackModeComp_;
  MonitorElement* emtfTrackBXComp_;
  MonitorElement* emtfTrackSectorIndexComp_;

};

#endif

    
  

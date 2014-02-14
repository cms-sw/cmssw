#ifndef RPCRecHitProbability_h
#define RPCRecHitProbability_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//DQMServices
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
///Data Format
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include<string>


class RPCRecHitProbability : public DQMEDAnalyzer {
 public:
  explicit RPCRecHitProbability( const edm::ParameterSet&);
  ~RPCRecHitProbability();
  
 protected:
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
   
 private:
  
  void makeDcsInfo(const edm::Event& ) ;
  
  std::string muonFolder_;
  int counter;
  
  bool dcs_;
  float muPtCut_, muEtaCut_;
  
  std::string globalFolder_;
  std::string subsystemFolder_;
  
  bool saveRootFile;
  std::string RootFileName;
  
  MonitorElement * NumberOfMuonPt_B_;
  MonitorElement * NumberOfMuonPhi_B_;
  MonitorElement * NumberOfMuonPt_EP_;
  MonitorElement * NumberOfMuonPhi_EP_;
  MonitorElement * NumberOfMuonPt_EM_;
  MonitorElement * NumberOfMuonPhi_EM_;
  MonitorElement * NumberOfMuonEta_;
  MonitorElement * RPCRecHitMuonEta_;
  
  MonitorElement * recHitEta_[6];
  MonitorElement * recHitPt_B_[6];
  MonitorElement * recHitPhi_B_[6];
  MonitorElement * recHitPt_EP_[6];
  MonitorElement * recHitPhi_EP_[6];
  MonitorElement * recHitPt_EM_[6];
  MonitorElement * recHitPhi_EM_[6];
  
  edm::EDGetTokenT<reco::CandidateView> muonLabel_;
  edm::EDGetTokenT<DcsStatusCollection> scalersRawToDigiLabel_;
  
};

#endif

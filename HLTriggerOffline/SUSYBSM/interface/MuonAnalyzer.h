#ifndef MuonAnalyzer_h
#define MuonAnalyzer_h

/*  \class 
*
*  Author: Philip Hebda
*
*/
// system include files
#include <memory>
#include <cassert>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
//#include "FWCore/messageLogger/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerAOD.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "TH1.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"


class MuonAnalyzerSBSM {

 public:
  MuonAnalyzerSBSM(edm::InputTag, edm::InputTag);
  virtual ~MuonAnalyzerSBSM(){};

  void InitializePlots(DQMStore *, const std::string);
  void FillPlots(const edm::Event&, const edm::EventSetup&);

 private:

  MonitorElement* hLeadRecoMuonPt_1_ByEvent;
  MonitorElement* hLeadRecoMuonEta_1_ByEvent;
  MonitorElement* hLeadRecoMuonPt_2_ByEvent;
  MonitorElement* hLeadRecoMuonEta_2_ByEvent;
  MonitorElement* hLeadRecoMuonPt_3_ByEvent;
  MonitorElement* hLeadRecoMuonEta_3_ByEvent;

  MonitorElement* hLeadAssocRecoMuonPt_1_ByEvent;
  MonitorElement* hLeadAssocRecoMuonEta_1_ByEvent;
  MonitorElement* hLeadAssocRecoMuonPt_2_ByEvent;
  MonitorElement* hLeadAssocRecoMuonEta_2_ByEvent;
  MonitorElement* hLeadAssocRecoMuonPt_3_ByEvent;
  MonitorElement* hLeadAssocRecoMuonEta_3_ByEvent;

  MonitorElement* hRecoMuonPt_1_ByMuon;
  MonitorElement* hRecoMuonEta_1_ByMuon;
  MonitorElement* hRecoMuonPt_2_ByMuon;
  MonitorElement* hRecoMuonEta_2_ByMuon;
  MonitorElement* hRecoMuonPt_3_ByMuon;
  MonitorElement* hRecoMuonEta_3_ByMuon;

  MonitorElement* hAssocRecoMuonPt_1_ByMuon;
  MonitorElement* hAssocRecoMuonEta_1_ByMuon;
  MonitorElement* hAssocRecoMuonPt_2_ByMuon;
  MonitorElement* hAssocRecoMuonEta_2_ByMuon;
  MonitorElement* hAssocRecoMuonPt_3_ByMuon;
  MonitorElement* hAssocRecoMuonEta_3_ByMuon;

  reco::MuonCollection Muons;
  edm::InputTag triggerTag_;
  edm::InputTag muonTag_;

  bool find(const std::vector<int>&, int);

};


#endif

#ifndef ZCounting_H
#define ZCounting_H

#include "FWCore/Framework/interface/MakerMacros.h"      // definitions for declaring plug-in modules
#include "FWCore/Framework/interface/Frameworkfwd.h"     // declaration of EDM types
#include "FWCore/Framework/interface/EDAnalyzer.h"       // EDAnalyzer class
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"  // Parameters
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>                                        // string class
#include <cassert>

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMOffline/Lumi/interface/TriggerDefs.h"
#include "DQMOffline/Lumi/interface/TTrigger.h"
#include "DQMOffline/Lumi/interface/TriggerTools.h"

class TFile;
class TH1D;
class TTree;
class TClonesArray;
namespace edm {
  class TriggerResults;
  class TriggerNames;
}
namespace ZCountingTrigger {
  class TTrigger;
}
 
class ZCounting: public DQMEDAnalyzer{

public:

  ZCounting(const edm::ParameterSet& ps);
  virtual ~ZCounting();

  enum MuonIDTypes  { NoneID, LooseID, MediumID, TightID };
  enum MuonIsoTypes { NoneIso, TrackerIso, PFIso };
  
protected:

  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;

private:
  //other functions
  bool isMuonTrigger(const ZCountingTrigger::TTrigger &triggerMenu, const TriggerBits &hltBits);
  bool isMuonTriggerObj(const ZCountingTrigger::TTrigger &triggerMenu, const TriggerObjects &hltMatchBits);
  bool passMuonID(const reco::Muon& muon, const reco::Vertex& vtx, const MuonIDTypes &idType);
  bool passMuonIso(const reco::Muon& muon, const MuonIsoTypes &isoType, const float isoCut);

  // initialization from HLT menu; needs to be called on every change in HLT menu
  void initHLT(const edm::TriggerResults&, const edm::TriggerNames&);

  // EDM object collection names
  edm::ParameterSetID fTriggerNamesID;
  edm::InputTag       fHLTObjTag;
  edm::InputTag       fHLTTag;
  edm::EDGetTokenT<trigger::TriggerEvent> fHLTObjTag_token;
  edm::EDGetTokenT<edm::TriggerResults> fHLTTag_token;
  std::string fPVName;
  edm::EDGetTokenT<reco::VertexCollection> fPVName_token;
  std::string fMuonName;
  edm::EDGetTokenT<reco::MuonCollection> fMuonName_token;
  std::string fTrackName;
  edm::EDGetTokenT<reco::TrackCollection> fTrackName_token;

  // bacon fillers
  std::unique_ptr<ZCountingTrigger::TTrigger> fTrigger;

  std::string  IDTypestr_;
  std::string  IsoTypestr_;
  MuonIDTypes  IDType_  {NoneID};
  MuonIsoTypes IsoType_ {NoneIso};
  double IsoCut_;

  double PtCutL1_;
  double PtCutL2_;
  double EtaCutL1_;
  double EtaCutL2_;

  int    MassBin_;
  double MassMin_;
  double MassMax_;

  int    LumiBin_;
  double LumiMin_;
  double LumiMax_;

  int    PVBin_;
  double PVMin_;
  double PVMax_;

  double VtxNTracksFitCut_;
  double VtxNdofCut_;
  double VtxAbsZCut_;
  double VtxRhoCut_;

  const double MUON_MASS  = 0.105658369;
  const double MUON_BOUND = 0.9;

  // Histograms
  MonitorElement* h_mass_HLT_pass_central;
  MonitorElement* h_mass_HLT_pass_forward;
  MonitorElement* h_mass_HLT_fail_central;
  MonitorElement* h_mass_HLT_fail_forward;

  MonitorElement* h_mass_SIT_pass_central;
  MonitorElement* h_mass_SIT_pass_forward;
  MonitorElement* h_mass_SIT_fail_central;
  MonitorElement* h_mass_SIT_fail_forward;

  MonitorElement* h_mass_Sta_pass_central;
  MonitorElement* h_mass_Sta_pass_forward;
  MonitorElement* h_mass_Sta_fail_central;
  MonitorElement* h_mass_Sta_fail_forward;

  MonitorElement* h_npv;
  MonitorElement* h_yield_Z;
};


#endif

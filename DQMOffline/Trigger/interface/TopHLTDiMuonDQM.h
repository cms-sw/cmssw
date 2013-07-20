#ifndef TopHLTDiMuonDQM_H
#define TopHLTDiMuonDQM_H

/*
 *  $Date: 2010/08/13 09:12:05 $
 *  $Revision: 1.9 $
 *  \author M. Marienfeld - DESY Hamburg
 */

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

class TopHLTDiMuonDQM : public edm::EDAnalyzer {

 public:

  TopHLTDiMuonDQM( const edm::ParameterSet& );
  ~TopHLTDiMuonDQM();

 protected:   

  void beginJob();
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

  void analyze(const edm::Event&, const edm::EventSetup&);

  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  void endRun(const edm::Run&, const edm::EventSetup&);
  void endJob();

 private:

  DQMStore * dbe_;
  std::string monitorName_;

  edm::InputTag triggerResults_;
  edm::InputTag triggerEvent_;
  edm::InputTag triggerFilter_;

  edm::InputTag vertex_;
  edm::InputTag muons_;

  std::vector<std::string> hltPaths_L1_;
  std::vector<std::string> hltPaths_L3_;
  std::vector<std::string> hltPaths_sig_;
  std::vector<std::string> hltPaths_trig_;

  double vertex_X_cut_;
  double vertex_Y_cut_;
  double vertex_Z_cut_;

  double muon_pT_cut_;
  double muon_eta_cut_;
  double muon_iso_cut_;

  double MassWindow_up_;
  double MassWindow_down_;

  MonitorElement * Trigs;
  MonitorElement * NTracks;
  MonitorElement * NMuons;
  MonitorElement * NMuons_charge;
  MonitorElement * NMuons_iso;
  MonitorElement * PtMuons;
  MonitorElement * PtMuons_LOGX;
  MonitorElement * EtaMuons;
  MonitorElement * PhiMuons;
  MonitorElement * CombRelIso03;
  MonitorElement * VxVy_muons;
  MonitorElement * Vz_muons;
  MonitorElement * PixelHits_muons;
  MonitorElement * TrackerHits_muons;

  MonitorElement * TriggerEfficiencies;
  MonitorElement * TriggerEfficiencies_sig;
  MonitorElement * TriggerEfficiencies_trig;

  MonitorElement * MuonEfficiency_pT;
  MonitorElement * MuonEfficiency_pT_sig;
  MonitorElement * MuonEfficiency_pT_trig;

  MonitorElement * MuonEfficiency_pT_LOGX;
  MonitorElement * MuonEfficiency_pT_LOGX_sig;
  MonitorElement * MuonEfficiency_pT_LOGX_trig;

  MonitorElement * MuonEfficiency_eta;
  MonitorElement * MuonEfficiency_eta_sig;
  MonitorElement * MuonEfficiency_eta_trig;

  MonitorElement * MuonEfficiency_phi;
  MonitorElement * MuonEfficiency_phi_sig;
  MonitorElement * MuonEfficiency_phi_trig;

  MonitorElement * DiMuonMassRC;
  MonitorElement * DiMuonMassWC;
  MonitorElement * DiMuonMassRC_LOGX;
  MonitorElement * DiMuonMassWC_LOGX;

  MonitorElement * DeltaEtaMuonsRC;
  MonitorElement * DeltaPhiMuonsRC;
  MonitorElement * DeltaEtaMuonsWC;
  MonitorElement * DeltaPhiMuonsWC;

  MonitorElement * DeltaR_Trig;
  MonitorElement * DeltaR_Reco;
  MonitorElement * DeltaR_Match;
  MonitorElement * Trigger_Match;

};

#endif

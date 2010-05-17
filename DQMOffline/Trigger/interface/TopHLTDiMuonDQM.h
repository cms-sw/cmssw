#ifndef TopHLTDiMuonDQM_H
#define TopHLTDiMuonDQM_H

/*
 *  DQM HLT Dimuon Test Client
 *
 *  $Date: 2010/03/02 17:29:11 $
 *  $Revision: 1.5 $
 *  \author  M. Vander Donckt CERN
 *   
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

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
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
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

//
// class declaration
//

class TopHLTDiMuonDQM : public edm::EDAnalyzer {

 public:

  TopHLTDiMuonDQM( const edm::ParameterSet& );
  ~TopHLTDiMuonDQM();

 protected:   

  void beginJob();
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

  void analyze(const edm::Event&, const edm::EventSetup&);

  void endLuminosityBlock(  const edm::LuminosityBlock&, const edm::EventSetup&);
  void endRun(  const edm::Run&, const edm::EventSetup&);
  void endJob();

 private:

  DQMStore* dbe_;
  std::string monitorName_;
  std::string level_;

  int N_sig[100];
  int N_trig[100];
  float Eff[100];

  edm::InputTag triggerResults_;
  edm::InputTag triggerEvent_;
  edm::InputTag triggerFilter_;

  edm::InputTag L1_Collection_;
  edm::InputTag L3_Collection_;
  edm::InputTag L3_Isolation_;
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

  // ----------member data ---------------------------

  MonitorElement * Trigs;
  MonitorElement * NTracks;
  MonitorElement * NMuons;
  MonitorElement * NMuons_charge;
  MonitorElement * NMuons_iso;
  MonitorElement * PtMuons;
  MonitorElement * PtMuons_sig;
  MonitorElement * PtMuons_trig;
  MonitorElement * EtaMuons;
  MonitorElement * EtaMuons_sig;
  MonitorElement * EtaMuons_trig;
  MonitorElement * PhiMuons;
  MonitorElement * CombRelIso03;
  MonitorElement * VxVy_muons;
  MonitorElement * Vz_muons;

  MonitorElement * DiMuonMassRC;
  MonitorElement * DiMuonMassWC;
  MonitorElement * DiMuonMassRC_LOGX;
  MonitorElement * DiMuonMassWC_LOGX;
  MonitorElement * DiMuonMassRC_LOG10;
  MonitorElement * DiMuonMassWC_LOG10;

  MonitorElement * DeltaEtaMuons;
  MonitorElement * DeltaPhiMuons;
  MonitorElement * MuonEfficiency_pT;
  MonitorElement * MuonEfficiency_eta;
  MonitorElement * TriggerEfficiencies;

};

#endif

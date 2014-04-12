#ifndef GenPurposeSkimmerData_H
#define GenPurposeSkimmerData_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// other files
// root + maths
#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#include "TVector.h"
#include "TString.h"
#include "TMath.h"
//
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerAOD.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
//
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"

//
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/Math/interface/deltaR.h"
//
// class decleration
//

class GenPurposeSkimmerData : public edm::EDAnalyzer {
   public:
      explicit GenPurposeSkimmerData(const edm::ParameterSet&);
      ~GenPurposeSkimmerData();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------


  std::string outputFile_;
  int tree_fills_;

  edm::EDGetTokenT<pat::ElectronCollection> ElectronCollectionToken_;
  edm::InputTag MCCollection_;
  edm::EDGetTokenT<reco::GenParticleCollection> MCCollectionToken_;
  edm::InputTag MetCollectionTag_;
  edm::EDGetTokenT<reco::CaloMETCollection> MetCollectionToken_;
  edm::EDGetTokenT<pat::METCollection> mcMetCollectionToken_;
  edm::EDGetTokenT<reco::METCollection> tcMetCollectionToken_;
  edm::EDGetTokenT<reco::PFMETCollection> pfMetCollectionToken_;
  edm::EDGetTokenT<pat::METCollection> t1MetCollectionToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMetCollectionToken_;
  //
  edm::InputTag HLTCollectionE29_;
  edm::EDGetTokenT<trigger::TriggerEvent> HLTCollectionE29Token_;
  edm::InputTag HLTCollectionE31_;
  edm::EDGetTokenT<trigger::TriggerEvent> HLTCollectionE31Token_;
  edm::InputTag HLTTriggerResultsE29_;
  edm::EDGetTokenT<edm::TriggerResults> HLTTriggerResultsE29Token_;
  edm::InputTag HLTTriggerResultsE31_;
  edm::EDGetTokenT<edm::TriggerResults> HLTTriggerResultsE31Token_;
  edm::InputTag HLTFilterType_[25];
  //  std::string HLTPath_[25];
  edm::EDGetTokenT<reco::TrackCollection> ctfTracksToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> corHybridscToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> multi5x5scToken_;
  edm::EDGetTokenT<reco::BeamSpot> offlineBeamSpotToken_;
  edm::EDGetTokenT<pat::MuonCollection> pMuonsToken_;

  TTree * probe_tree;
  TFile * histofile;
  //


  //probe SC variables
  double probe_sc_eta_for_tree[4];
  double probe_sc_phi_for_tree[4];
  double probe_sc_et_for_tree[4];
  int probe_sc_pass_fiducial_cut[4];
  int probe_sc_pass_et_cut[4];

  //probe electron variables
  double probe_ele_eta_for_tree[4];
  double probe_ele_phi_for_tree[4];
  double probe_ele_et_for_tree[4];
  double probe_ele_Xvertex_for_tree[4];
  double probe_ele_Yvertex_for_tree[4];
  double probe_ele_Zvertex_for_tree[4];
  double probe_ele_tip[4];
  int probe_charge_for_tree[4];
  int probe_index_for_tree[4];

  //efficiency cuts
  int probe_ele_pass_fiducial_cut[4];
  int probe_ele_pass_et_cut[4];
  int probe_pass_recoEle_cut[4];
  int probe_pass_iso_cut[4];
  //
  double probe_isolation_value[4];
  double probe_iso_user[4];
  //
  double probe_ecal_isolation_value[4];
  double probe_ecal_iso_user[4];

  double probe_hcal_isolation_value[4];
  double probe_hcal_iso_user[4];
  //
  int probe_classification_index_for_tree[4];
  int probe_pass_tip_cut[4];
  //
  int probe_pass_id_robust_loose[4];
  int probe_pass_id_robust_tight[4];
  int probe_pass_id_loose[4];
  int probe_pass_id_tight[4];
  double probe_ele_hoe[4];
  double probe_ele_shh[4];
  double probe_ele_sihih[4];
  double probe_ele_dhi[4];
  double probe_ele_dfi[4];
  double probe_ele_eop[4];
  double probe_ele_pin[4];
  double probe_ele_pout[4];
  double probe_ele_e5x5[4];
  double probe_ele_e2x5[4];
  double probe_ele_e1x5[4];

  //
  int probe_pass_trigger_cut[4][25];
  double probe_hlt_matched_dr[4];
  //
  double MCMatch_Deta_;
  double MCMatch_Dphi_;
  int probe_mc_matched[4];
  double probe_mc_matched_deta[4];
  double probe_mc_matched_dphi[4];
  double probe_mc_matched_denergy[4];
  int probe_mc_matched_mother[4];

 //event variables
  int numberOfHLTFilterObjects[25];

  int event_HLTPath[25];

  double BarrelMaxEta;
  double EndcapMinEta;
  double EndcapMaxEta;

  double ProbeSCMinEt;
  double ProbeRecoEleSCMaxDE;

  double ProbeHLTObjMaxDR;
  double RecoEleSeedBCMaxDE;
  double GsfTrackMinInnerPt;

  int elec_number_in_event;
  int elec_1_duplicate_removal;

  double    event_MET, event_MET_sig;
  //  double    event_MET_eta;
  double    event_MET_phi;
  double    event_tcMET, event_tcMET_sig, event_tcMET_phi;
  double    event_pfMET, event_pfMET_sig;
  //  double    event_pfMET_eta;
  double    event_pfMET_phi;
  //double    event_genMET, event_genMET_sig;
  //  double    event_genMET_eta;
  //double    event_genMET_phi;
  //
  double event_t1MET, event_t1MET_phi, event_t1MET_sig;
  double event_mcMET, event_mcMET_phi, event_mcMET_sig;
  //
  //
  //
  double sc_hybrid_et[5], sc_hybrid_eta[5], sc_hybrid_phi[5];
  double sc_multi5x5_et[5], sc_multi5x5_eta[5], sc_multi5x5_phi[5];
  //
  double ctf_track_pt[20], ctf_track_eta[20], ctf_track_phi[20];
  double ctf_track_vx[20], ctf_track_vy[20], ctf_track_vz[20];
  double ctf_track_tip[20],  ctf_track_tip_bs[20];
  //
  double muon_pt[4], muon_eta[4], muon_phi[4];
  double muon_vx[4], muon_vy[4], muon_vz[4];
  double muon_tip[4],  muon_tip_bs[4];
};


#endif


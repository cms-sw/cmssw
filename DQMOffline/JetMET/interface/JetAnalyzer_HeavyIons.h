#ifndef JetAnalyzer_HeavyIons_H
#define JetAnalyzer_HeavyIons_H

//
// Jet Tester class for heavy ion jets. for DQM jet analysis monitoring
// For CMSSW_7_4_X, especially reading background subtracted jets
// author: Raghav Kunnawalkam Elayavalli,
//         Jan 12th 2015
//         Rutgers University, email: raghav.k.e at CERN dot CH
//
// this class will be very similar to the class available in the validation suite under RecoJets/JetTester_HeavyIons
//

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateWithRef.h"

// include the basic jet for the PuPF jets.
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
// include the pf candidates
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
// include the voronoi subtraction
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"
// include the centrality variables
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include <map>
#include <string>

const Int_t MAXPARTICLE = 10000;
const Double_t BarrelEta = 2.0;
const Double_t EndcapEta = 3.0;
const Double_t ForwardEta = 5.0;

class JetAnalyzer_HeavyIons : public DQMEDAnalyzer {
public:
  explicit JetAnalyzer_HeavyIons(const edm::ParameterSet&);
  ~JetAnalyzer_HeavyIons() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  static constexpr int fourierOrder_ = 5;
  static constexpr int etaBins_ = 7;
  static constexpr int ptBins_ = 7;

  //default values - these are changed by the etaMap values for the CS plots
  const double edge_pseudorapidity[etaBins_ + 1] = {-5, -3, -2.1, -1.3, 1.3, 2.1, 3, 5};
  const int ptBin[ptBins_ + 1] = {0, 20, 40, 60, 100, 150, 300, 99999};

  static constexpr int nedge_pseudorapidity = etaBins_ + 1;

  edm::InputTag mInputCollection;
  edm::InputTag mInputVtxCollection;
  edm::InputTag mInputPFCandCollection;
  edm::InputTag mInputCsCandCollection;

  std::string mOutputFile;
  std::string JetType;
  std::string UEAlgo;
  edm::InputTag Background;
  double mRecoJetPtThreshold;
  double mReverseEnergyFractionThreshold;
  double mRThreshold;
  std::string JetCorrectionService;

  edm::InputTag centralityTag_;
  edm::EDGetTokenT<reco::Centrality> centralityToken;
  edm::Handle<reco::Centrality> centrality_;

  edm::InputTag centralityBinTag_;
  edm::EDGetTokenT<int> centralityBinToken;
  edm::Handle<int> centralityBin_;

  //Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
  edm::EDGetTokenT<CaloTowerCollection> caloTowersToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::BasicJetCollection> basicJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection> jptJetsToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> csCandToken_;
  edm::EDGetTokenT<reco::CandidateView> pfCandViewToken_;
  edm::EDGetTokenT<reco::CandidateView> caloCandViewToken_;

  edm::EDGetTokenT<std::vector<reco::Vertex>> hiVertexToken_;

  edm::EDGetTokenT<std::vector<double>> etaToken_;
  edm::EDGetTokenT<std::vector<double>> rhoToken_;
  edm::EDGetTokenT<std::vector<double>> rhomToken_;

  MonitorElement* mNPFpart;
  MonitorElement* mPFPt;
  MonitorElement* mPFEta;
  MonitorElement* mPFPhi;

  MonitorElement* mPFArea;
  MonitorElement* mPFDeltaR;           //MZ
  MonitorElement* mPFDeltaR_Scaled_R;  //MZ

  MonitorElement* mNCalopart;
  MonitorElement* mCaloPt;
  MonitorElement* mCaloEta;
  MonitorElement* mCaloPhi;
  MonitorElement* mCaloArea;
  MonitorElement* mSumpt;
  MonitorElement* mSumPFPt;
  MonitorElement* mSumPFPt_eta;

  MonitorElement* mSumCaloPt;
  MonitorElement* mSumCaloPt_eta;

  MonitorElement* mSumSquaredPFPt;
  MonitorElement* mSumSquaredPFPt_eta;

  MonitorElement* mSumSquaredCaloPt;
  MonitorElement* mSumSquaredCaloPt_eta;

  // Event variables (including centrality)
  MonitorElement* mNvtx;
  MonitorElement* mHF;

  // new additions Jan 12th 2015
  MonitorElement* mSumPFPt_HF;
  MonitorElement* mSumCaloPt_HF;

  MonitorElement* mSumPFPtEtaDep[etaBins_];
  MonitorElement* mSumCaloPtEtaDep[etaBins_];

  // Jet parameters
  MonitorElement* mEta;
  MonitorElement* mPhi;
  MonitorElement* mPt;
  MonitorElement* mP;
  MonitorElement* mEnergy;
  MonitorElement* mMass;
  MonitorElement* mConstituents;
  MonitorElement* mJetArea;
  MonitorElement* mjetpileup;
  MonitorElement* mNJets;
  MonitorElement* mNJets_40;

  MonitorElement* mRhoDist_vsEta;
  MonitorElement* mRhoMDist_vsEta;
  MonitorElement* mRhoDist_vsPt;
  MonitorElement* mRhoMDist_vsPt;
  MonitorElement* mRhoDist_vsCent[etaBins_];
  MonitorElement* mRhoMDist_vsCent[etaBins_];
  MonitorElement* rhoEtaRange;
  MonitorElement* mCSCandpT_vsPt[etaBins_];
  MonitorElement* mCSCand_corrPFcand[etaBins_];
  MonitorElement* mSubtractedEFrac[ptBins_][etaBins_];
  MonitorElement* mSubtractedE[ptBins_][etaBins_];

  MonitorElement* mPFCandpT_vs_eta_Unknown;        // pf id 0
  MonitorElement* mPFCandpT_vs_eta_ChargedHadron;  // pf id - 1
  MonitorElement* mPFCandpT_vs_eta_electron;       // pf id - 2
  MonitorElement* mPFCandpT_vs_eta_muon;           // pf id - 3
  MonitorElement* mPFCandpT_vs_eta_photon;         // pf id - 4
  MonitorElement* mPFCandpT_vs_eta_NeutralHadron;  // pf id - 5
  MonitorElement* mPFCandpT_vs_eta_HadE_inHF;      // pf id - 6
  MonitorElement* mPFCandpT_vs_eta_EME_inHF;       // pf id - 7

  MonitorElement* mPFCandpT_Barrel_Unknown;        // pf id 0
  MonitorElement* mPFCandpT_Barrel_ChargedHadron;  // pf id - 1
  MonitorElement* mPFCandpT_Barrel_electron;       // pf id - 2
  MonitorElement* mPFCandpT_Barrel_muon;           // pf id - 3
  MonitorElement* mPFCandpT_Barrel_photon;         // pf id - 4
  MonitorElement* mPFCandpT_Barrel_NeutralHadron;  // pf id - 5
  MonitorElement* mPFCandpT_Barrel_HadE_inHF;      // pf id - 6
  MonitorElement* mPFCandpT_Barrel_EME_inHF;       // pf id - 7

  MonitorElement* mPFCandpT_Endcap_Unknown;        // pf id 0
  MonitorElement* mPFCandpT_Endcap_ChargedHadron;  // pf id - 1
  MonitorElement* mPFCandpT_Endcap_electron;       // pf id - 2
  MonitorElement* mPFCandpT_Endcap_muon;           // pf id - 3
  MonitorElement* mPFCandpT_Endcap_photon;         // pf id - 4
  MonitorElement* mPFCandpT_Endcap_NeutralHadron;  // pf id - 5
  MonitorElement* mPFCandpT_Endcap_HadE_inHF;      // pf id - 6
  MonitorElement* mPFCandpT_Endcap_EME_inHF;       // pf id - 7

  MonitorElement* mPFCandpT_Forward_Unknown;        // pf id 0
  MonitorElement* mPFCandpT_Forward_ChargedHadron;  // pf id - 1
  MonitorElement* mPFCandpT_Forward_electron;       // pf id - 2
  MonitorElement* mPFCandpT_Forward_muon;           // pf id - 3
  MonitorElement* mPFCandpT_Forward_photon;         // pf id - 4
  MonitorElement* mPFCandpT_Forward_NeutralHadron;  // pf id - 5
  MonitorElement* mPFCandpT_Forward_HadE_inHF;      // pf id - 6
  MonitorElement* mPFCandpT_Forward_EME_inHF;       // pf id - 7

  // Parameters

  bool isCaloJet;
  bool isJPTJet;
  bool isPFJet;
};

#endif

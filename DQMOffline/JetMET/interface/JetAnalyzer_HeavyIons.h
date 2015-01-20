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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"

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

class MonitorElement;

class JetAnalyzer_HeavyIons : public DQMEDAnalyzer {
 public:

  explicit JetAnalyzer_HeavyIons (const edm::ParameterSet&);
  virtual ~JetAnalyzer_HeavyIons();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&); 
  virtual void beginJob();
  virtual void endJob();
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) ;
  
 private:
  
  edm::InputTag   mInputCollection;
  edm::InputTag   mInputPFCandCollection;
  edm::InputTag   centrality;
  
  std::string     mOutputFile;
  std::string     JetType;
  std::string     UEAlgo;
  edm::InputTag   Background;
  double          mRecoJetPtThreshold;
  double          mReverseEnergyFractionThreshold;
  double          mRThreshold;
  std::string     JetCorrectionService;

  //Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex> > pvToken_;
  edm::EDGetTokenT<CaloTowerCollection > caloTowersToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::BasicJetCollection> basicJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection> jptJetsToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_; 
  edm::EDGetTokenT<reco::CandidateView> pfCandViewToken_;
  edm::EDGetTokenT<reco::CandidateView> caloCandViewToken_;

  //edm::EDGetTokenT<reco::VoronoiMap> backgrounds_;
  edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground>> backgrounds_;
  edm::EDGetTokenT<std::vector<float>> backgrounds_value_;
  edm::EDGetTokenT<reco::Centrality> centralityToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hiVertexToken_;

  //Include Particle flow variables 
  MonitorElement *mNPFpart;
  MonitorElement *mPFPt;
  MonitorElement *mPFEta;
  MonitorElement *mPFPhi;
  MonitorElement *mPFVsPt;
  MonitorElement *mPFVsPtInitial;
  MonitorElement *mPFVsPtEqualized;
  MonitorElement *mPFArea;
  MonitorElement *mNCalopart;
  MonitorElement *mCaloPt;
  MonitorElement *mCaloEta;
  MonitorElement *mCaloPhi;
  MonitorElement *mCaloVsPt;
  MonitorElement *mCaloVsPtInitial;
  MonitorElement *mCaloVsPtEqualized;
  MonitorElement *mCaloArea;
  MonitorElement *mSumpt;
  MonitorElement *mvn;
  MonitorElement *mpsin;
  // MonitorElement *ueraw;  

  // necessary plots for the vs validation which is the vn weighted SumpT for differnet eta bins, 
  MonitorElement *mSumPFVsPt;
  MonitorElement *mSumPFVsPtInitial;
  MonitorElement *mSumPFPt;

  MonitorElement *mSumPFVsPtInitial_eta;
  MonitorElement *mSumPFVsPt_eta;
  MonitorElement *mSumPFPt_eta;

  MonitorElement *mSumCaloVsPt;
  MonitorElement *mSumCaloVsPtInitial;
  MonitorElement *mSumCaloPt;

  MonitorElement *mSumCaloVsPtInitial_eta;
  MonitorElement *mSumCaloVsPt_eta;
  MonitorElement *mSumCaloPt_eta;

  // Event variables (including centrality)
  MonitorElement* mNvtx;
  MonitorElement* mHF;

  // new additions Jan 12th 2015
  MonitorElement *mSumPFVsPt_HF;
  MonitorElement *mSumPFVsPtInitial_HF;
  MonitorElement *mSumPFPt_HF;
  MonitorElement *mPFVsPtInitial_eta_phi;
  MonitorElement *mPFVsPt_eta_phi;
  MonitorElement *mPFPt_eta_phi;
  //MonitorElement *mSumDeltapT_HF;
  MonitorElement *mDeltapT;
  MonitorElement *mDeltapT_eta;
  //MonitorElement *mDeltapT_phiMinusPsi2;
  MonitorElement *mDeltapT_eta_phi;

  MonitorElement *mSumCaloVsPt_HF;
  MonitorElement *mSumCaloVsPtInitial_HF;
  MonitorElement *mSumCaloPt_HF;
  MonitorElement *mCaloVsPtInitial_eta_phi;
  MonitorElement *mCaloVsPt_eta_phi;
  MonitorElement *mCaloPt_eta_phi;

  MonitorElement *mVs_0_x;
  MonitorElement *mVs_0_y;
  MonitorElement *mVs_1_x;
  MonitorElement *mVs_1_y;
  MonitorElement *mVs_2_x;
  MonitorElement *mVs_2_y;
  MonitorElement *mVs_0_x_versus_HF;
  MonitorElement *mVs_0_y_versus_HF;
  MonitorElement *mVs_1_x_versus_HF;
  MonitorElement *mVs_1_y_versus_HF;
  MonitorElement *mVs_2_x_versus_HF;
  MonitorElement *mVs_2_y_versus_HF;
  
  MonitorElement *mPFVsPtInitial_n5p191_n2p650;
  MonitorElement *mPFVsPtInitial_n2p650_n2p043;
  MonitorElement *mPFVsPtInitial_n2p043_n1p740;
  MonitorElement *mPFVsPtInitial_n1p740_n1p479;
  MonitorElement *mPFVsPtInitial_n1p479_n1p131;
  MonitorElement *mPFVsPtInitial_n1p131_n0p783;
  MonitorElement *mPFVsPtInitial_n0p783_n0p522;
  MonitorElement *mPFVsPtInitial_n0p522_0p522;
  MonitorElement *mPFVsPtInitial_0p522_0p783;
  MonitorElement *mPFVsPtInitial_0p783_1p131;
  MonitorElement *mPFVsPtInitial_1p131_1p479;
  MonitorElement *mPFVsPtInitial_1p479_1p740;
  MonitorElement *mPFVsPtInitial_1p740_2p043;
  MonitorElement *mPFVsPtInitial_2p043_2p650;
  MonitorElement *mPFVsPtInitial_2p650_5p191;

  MonitorElement *mPFVsPt_n5p191_n2p650;
  MonitorElement *mPFVsPt_n2p650_n2p043;
  MonitorElement *mPFVsPt_n2p043_n1p740;
  MonitorElement *mPFVsPt_n1p740_n1p479;
  MonitorElement *mPFVsPt_n1p479_n1p131;
  MonitorElement *mPFVsPt_n1p131_n0p783;
  MonitorElement *mPFVsPt_n0p783_n0p522;
  MonitorElement *mPFVsPt_n0p522_0p522;
  MonitorElement *mPFVsPt_0p522_0p783;
  MonitorElement *mPFVsPt_0p783_1p131;
  MonitorElement *mPFVsPt_1p131_1p479;
  MonitorElement *mPFVsPt_1p479_1p740;
  MonitorElement *mPFVsPt_1p740_2p043;
  MonitorElement *mPFVsPt_2p043_2p650;
  MonitorElement *mPFVsPt_2p650_5p191;

  MonitorElement *mPFPt_n5p191_n2p650;
  MonitorElement *mPFPt_n2p650_n2p043;
  MonitorElement *mPFPt_n2p043_n1p740;
  MonitorElement *mPFPt_n1p740_n1p479;
  MonitorElement *mPFPt_n1p479_n1p131;
  MonitorElement *mPFPt_n1p131_n0p783;
  MonitorElement *mPFPt_n0p783_n0p522;
  MonitorElement *mPFPt_n0p522_0p522;
  MonitorElement *mPFPt_0p522_0p783;
  MonitorElement *mPFPt_0p783_1p131;
  MonitorElement *mPFPt_1p131_1p479;
  MonitorElement *mPFPt_1p479_1p740;
  MonitorElement *mPFPt_1p740_2p043;
  MonitorElement *mPFPt_2p043_2p650;
  MonitorElement *mPFPt_2p650_5p191;
  
  
  MonitorElement *mCaloVsPtInitial_n5p191_n2p650;
  MonitorElement *mCaloVsPtInitial_n2p650_n2p043;
  MonitorElement *mCaloVsPtInitial_n2p043_n1p740;
  MonitorElement *mCaloVsPtInitial_n1p740_n1p479;
  MonitorElement *mCaloVsPtInitial_n1p479_n1p131;
  MonitorElement *mCaloVsPtInitial_n1p131_n0p783;
  MonitorElement *mCaloVsPtInitial_n0p783_n0p522;
  MonitorElement *mCaloVsPtInitial_n0p522_0p522;
  MonitorElement *mCaloVsPtInitial_0p522_0p783;
  MonitorElement *mCaloVsPtInitial_0p783_1p131;
  MonitorElement *mCaloVsPtInitial_1p131_1p479;
  MonitorElement *mCaloVsPtInitial_1p479_1p740;
  MonitorElement *mCaloVsPtInitial_1p740_2p043;
  MonitorElement *mCaloVsPtInitial_2p043_2p650;
  MonitorElement *mCaloVsPtInitial_2p650_5p191;

  MonitorElement *mCaloVsPt_n5p191_n2p650;
  MonitorElement *mCaloVsPt_n2p650_n2p043;
  MonitorElement *mCaloVsPt_n2p043_n1p740;
  MonitorElement *mCaloVsPt_n1p740_n1p479;
  MonitorElement *mCaloVsPt_n1p479_n1p131;
  MonitorElement *mCaloVsPt_n1p131_n0p783;
  MonitorElement *mCaloVsPt_n0p783_n0p522;
  MonitorElement *mCaloVsPt_n0p522_0p522;
  MonitorElement *mCaloVsPt_0p522_0p783;
  MonitorElement *mCaloVsPt_0p783_1p131;
  MonitorElement *mCaloVsPt_1p131_1p479;
  MonitorElement *mCaloVsPt_1p479_1p740;
  MonitorElement *mCaloVsPt_1p740_2p043;
  MonitorElement *mCaloVsPt_2p043_2p650;
  MonitorElement *mCaloVsPt_2p650_5p191;

  MonitorElement *mCaloPt_n5p191_n2p650;
  MonitorElement *mCaloPt_n2p650_n2p043;
  MonitorElement *mCaloPt_n2p043_n1p740;
  MonitorElement *mCaloPt_n1p740_n1p479;
  MonitorElement *mCaloPt_n1p479_n1p131;
  MonitorElement *mCaloPt_n1p131_n0p783;
  MonitorElement *mCaloPt_n0p783_n0p522;
  MonitorElement *mCaloPt_n0p522_0p522;
  MonitorElement *mCaloPt_0p522_0p783;
  MonitorElement *mCaloPt_0p783_1p131;
  MonitorElement *mCaloPt_1p131_1p479;
  MonitorElement *mCaloPt_1p479_1p740;
  MonitorElement *mCaloPt_1p740_2p043;
  MonitorElement *mCaloPt_2p043_2p650;
  MonitorElement *mCaloPt_2p650_5p191;
  
  
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

  // Parameters

  bool            isCaloJet;
  bool            isJPTJet;
  bool            isPFJet;

  static const Int_t fourierOrder_ = 5;
  static const Int_t etaBins_ = 15;

  static const size_t nedge_pseudorapidity = etaBins_ + 1;

};


#endif 

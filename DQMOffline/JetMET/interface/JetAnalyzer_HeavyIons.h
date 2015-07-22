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

class MonitorElement;

class JetAnalyzer_HeavyIons : public DQMEDAnalyzer {

 public:

  explicit JetAnalyzer_HeavyIons (const edm::ParameterSet&);
  virtual ~JetAnalyzer_HeavyIons();
  
  void analyze(const edm::Event&, const edm::EventSetup&) override; 
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  
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

  edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground>> backgrounds_;
  edm::EDGetTokenT<std::vector<float>> backgrounds_value_;
  edm::EDGetTokenT<reco::Centrality> centralityToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hiVertexToken_;

  MonitorElement *mNPFpart;
  MonitorElement *mPFPt;
  MonitorElement *mPFEta;
  MonitorElement *mPFPhi;
  MonitorElement *mPFVsPt;
  MonitorElement *mPFVsPtInitial;
  MonitorElement *mPFArea;
  MonitorElement *mNCalopart;
  MonitorElement *mCaloPt;
  MonitorElement *mCaloEta;
  MonitorElement *mCaloPhi;
  MonitorElement *mCaloVsPt;
  MonitorElement *mCaloVsPtInitial;
  MonitorElement *mCaloArea;
  MonitorElement *mSumpt;
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

  MonitorElement *mSumSquaredPFVsPt;
  MonitorElement *mSumSquaredPFVsPtInitial;
  MonitorElement *mSumSquaredPFPt;

  MonitorElement *mSumSquaredPFVsPtInitial_eta;
  MonitorElement *mSumSquaredPFVsPt_eta;
  MonitorElement *mSumSquaredPFPt_eta;

  MonitorElement *mSumSquaredCaloVsPt;
  MonitorElement *mSumSquaredCaloVsPtInitial;
  MonitorElement *mSumSquaredCaloPt;

  MonitorElement *mSumSquaredCaloVsPtInitial_eta;
  MonitorElement *mSumSquaredCaloVsPt_eta;
  MonitorElement *mSumSquaredCaloPt_eta;  

  // Event variables (including centrality)
  MonitorElement* mNvtx;
  MonitorElement* mHF;

  // new additions Jan 12th 2015
  MonitorElement *mSumPFVsPt_HF;
  MonitorElement *mSumPFVsPtInitial_HF;
  MonitorElement *mSumPFPt_HF;

  MonitorElement *mSumCaloVsPt_HF;
  MonitorElement *mSumCaloVsPtInitial_HF;
  MonitorElement *mSumCaloPt_HF;
  
  MonitorElement *mSumPFVsPtInitial_n5p191_n2p650;
  MonitorElement *mSumPFVsPtInitial_n2p650_n2p043;
  MonitorElement *mSumPFVsPtInitial_n2p043_n1p740;
  MonitorElement *mSumPFVsPtInitial_n1p740_n1p479;
  MonitorElement *mSumPFVsPtInitial_n1p479_n1p131;
  MonitorElement *mSumPFVsPtInitial_n1p131_n0p783;
  MonitorElement *mSumPFVsPtInitial_n0p783_n0p522;
  MonitorElement *mSumPFVsPtInitial_n0p522_0p522;
  MonitorElement *mSumPFVsPtInitial_0p522_0p783;
  MonitorElement *mSumPFVsPtInitial_0p783_1p131;
  MonitorElement *mSumPFVsPtInitial_1p131_1p479;
  MonitorElement *mSumPFVsPtInitial_1p479_1p740;
  MonitorElement *mSumPFVsPtInitial_1p740_2p043;
  MonitorElement *mSumPFVsPtInitial_2p043_2p650;
  MonitorElement *mSumPFVsPtInitial_2p650_5p191;

  MonitorElement *mSumPFVsPt_n5p191_n2p650;
  MonitorElement *mSumPFVsPt_n2p650_n2p043;
  MonitorElement *mSumPFVsPt_n2p043_n1p740;
  MonitorElement *mSumPFVsPt_n1p740_n1p479;
  MonitorElement *mSumPFVsPt_n1p479_n1p131;
  MonitorElement *mSumPFVsPt_n1p131_n0p783;
  MonitorElement *mSumPFVsPt_n0p783_n0p522;
  MonitorElement *mSumPFVsPt_n0p522_0p522;
  MonitorElement *mSumPFVsPt_0p522_0p783;
  MonitorElement *mSumPFVsPt_0p783_1p131;
  MonitorElement *mSumPFVsPt_1p131_1p479;
  MonitorElement *mSumPFVsPt_1p479_1p740;
  MonitorElement *mSumPFVsPt_1p740_2p043;
  MonitorElement *mSumPFVsPt_2p043_2p650;
  MonitorElement *mSumPFVsPt_2p650_5p191;

  MonitorElement *mSumPFPt_n5p191_n2p650;
  MonitorElement *mSumPFPt_n2p650_n2p043;
  MonitorElement *mSumPFPt_n2p043_n1p740;
  MonitorElement *mSumPFPt_n1p740_n1p479;
  MonitorElement *mSumPFPt_n1p479_n1p131;
  MonitorElement *mSumPFPt_n1p131_n0p783;
  MonitorElement *mSumPFPt_n0p783_n0p522;
  MonitorElement *mSumPFPt_n0p522_0p522;
  MonitorElement *mSumPFPt_0p522_0p783;
  MonitorElement *mSumPFPt_0p783_1p131;
  MonitorElement *mSumPFPt_1p131_1p479;
  MonitorElement *mSumPFPt_1p479_1p740;
  MonitorElement *mSumPFPt_1p740_2p043;
  MonitorElement *mSumPFPt_2p043_2p650;
  MonitorElement *mSumPFPt_2p650_5p191;
  
 
  MonitorElement *mSumCaloVsPtInitial_n5p191_n2p650;
  MonitorElement *mSumCaloVsPtInitial_n2p650_n2p043;
  MonitorElement *mSumCaloVsPtInitial_n2p043_n1p740;
  MonitorElement *mSumCaloVsPtInitial_n1p740_n1p479;
  MonitorElement *mSumCaloVsPtInitial_n1p479_n1p131;
  MonitorElement *mSumCaloVsPtInitial_n1p131_n0p783;
  MonitorElement *mSumCaloVsPtInitial_n0p783_n0p522;
  MonitorElement *mSumCaloVsPtInitial_n0p522_0p522;
  MonitorElement *mSumCaloVsPtInitial_0p522_0p783;
  MonitorElement *mSumCaloVsPtInitial_0p783_1p131;
  MonitorElement *mSumCaloVsPtInitial_1p131_1p479;
  MonitorElement *mSumCaloVsPtInitial_1p479_1p740;
  MonitorElement *mSumCaloVsPtInitial_1p740_2p043;
  MonitorElement *mSumCaloVsPtInitial_2p043_2p650;
  MonitorElement *mSumCaloVsPtInitial_2p650_5p191;

  MonitorElement *mSumCaloVsPt_n5p191_n2p650;
  MonitorElement *mSumCaloVsPt_n2p650_n2p043;
  MonitorElement *mSumCaloVsPt_n2p043_n1p740;
  MonitorElement *mSumCaloVsPt_n1p740_n1p479;
  MonitorElement *mSumCaloVsPt_n1p479_n1p131;
  MonitorElement *mSumCaloVsPt_n1p131_n0p783;
  MonitorElement *mSumCaloVsPt_n0p783_n0p522;
  MonitorElement *mSumCaloVsPt_n0p522_0p522;
  MonitorElement *mSumCaloVsPt_0p522_0p783;
  MonitorElement *mSumCaloVsPt_0p783_1p131;
  MonitorElement *mSumCaloVsPt_1p131_1p479;
  MonitorElement *mSumCaloVsPt_1p479_1p740;
  MonitorElement *mSumCaloVsPt_1p740_2p043;
  MonitorElement *mSumCaloVsPt_2p043_2p650;
  MonitorElement *mSumCaloVsPt_2p650_5p191;

  MonitorElement *mSumCaloPt_n5p191_n2p650;
  MonitorElement *mSumCaloPt_n2p650_n2p043;
  MonitorElement *mSumCaloPt_n2p043_n1p740;
  MonitorElement *mSumCaloPt_n1p740_n1p479;
  MonitorElement *mSumCaloPt_n1p479_n1p131;
  MonitorElement *mSumCaloPt_n1p131_n0p783;
  MonitorElement *mSumCaloPt_n0p783_n0p522;
  MonitorElement *mSumCaloPt_n0p522_0p522;
  MonitorElement *mSumCaloPt_0p522_0p783;
  MonitorElement *mSumCaloPt_0p783_1p131;
  MonitorElement *mSumCaloPt_1p131_1p479;
  MonitorElement *mSumCaloPt_1p479_1p740;
  MonitorElement *mSumCaloPt_1p740_2p043;
  MonitorElement *mSumCaloPt_2p043_2p650;
  MonitorElement *mSumCaloPt_2p650_5p191;
  
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

  static const Int_t nedge_pseudorapidity = etaBins_ + 1;


};


#endif 

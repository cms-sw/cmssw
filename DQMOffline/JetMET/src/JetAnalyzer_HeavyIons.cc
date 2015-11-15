//
// Jet Analyzer class for heavy ion jets. for DQM jet analysis monitoring 
// For CMSSW_7_4_X, especially reading background subtracted jets 
// author: Raghav Kunnawalkam Elayavalli, Mohammed Zakaria (co Author)
//         Jan 12th 2015 
//         Rutgers University, email: raghav.k.e at CERN dot CH 
//         UIC, email: mzakaria @ CERN dot CH


#include "DQMOffline/JetMET/interface/JetAnalyzer_HeavyIons.h"

using namespace edm;
using namespace reco;
using namespace std;

// declare the constructors:

JetAnalyzer_HeavyIons::JetAnalyzer_HeavyIons(const edm::ParameterSet& iConfig) :
  mInputCollection               (iConfig.getParameter<edm::InputTag>       ("src")),
  mInputPFCandCollection         (iConfig.getParameter<edm::InputTag>       ("PFcands")),
  mOutputFile                    (iConfig.getUntrackedParameter<std::string>("OutputFile","")),
  JetType                        (iConfig.getUntrackedParameter<std::string>("JetType")),
  UEAlgo                         (iConfig.getUntrackedParameter<std::string>("UEAlgo")),
  Background                     (iConfig.getParameter<edm::InputTag>       ("Background")),
  mRecoJetPtThreshold            (iConfig.getParameter<double>              ("recoJetPtThreshold")),
  mReverseEnergyFractionThreshold(iConfig.getParameter<double>              ("reverseEnergyFractionThreshold")),
  mRThreshold                    (iConfig.getParameter<double>              ("RThreshold")),
  JetCorrectionService           (iConfig.getParameter<std::string>         ("JetCorrections"))
{
  std::string inputCollectionLabel(mInputCollection.label());

  isCaloJet = (std::string("calo")==JetType);
  isJPTJet  = (std::string("jpt") ==JetType);
  isPFJet   = (std::string("pf")  ==JetType);

  //consumes
  pvToken_ = consumes<std::vector<reco::Vertex> >(edm::InputTag("offlinePrimaryVertices"));
  caloTowersToken_ = consumes<CaloTowerCollection>(edm::InputTag("towerMaker"));
  if (isCaloJet) caloJetsToken_  = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isJPTJet)  jptJetsToken_   = consumes<reco::JPTJetCollection>(mInputCollection);
  if (isPFJet)   {
    if(std::string("Pu")==UEAlgo) basicJetsToken_    = consumes<reco::BasicJetCollection>(mInputCollection);
    if(std::string("Vs")==UEAlgo) pfJetsToken_    = consumes<reco::PFJetCollection>(mInputCollection);
  }

  pfCandToken_ = consumes<reco::PFCandidateCollection>(mInputPFCandCollection);
  pfCandViewToken_ = consumes<reco::CandidateView>(mInputPFCandCollection);
  caloCandViewToken_ = consumes<reco::CandidateView>(edm::InputTag("towerMaker"));
  backgrounds_ = consumes<edm::ValueMap<reco::VoronoiBackground> >(Background);  
  backgrounds_value_ = consumes<std::vector<float> >(Background);
                                  
  centralityTag_ = iConfig.getParameter<InputTag>("centralitycollection");
  centralityToken = consumes<reco::Centrality>(centralityTag_);

  centralityBinTag_ = (iConfig.getParameter<edm::InputTag> ("centralitybincollection"));
  centralityBinToken = consumes<int>(centralityBinTag_);

  hiVertexToken_ = consumes<std::vector<reco::Vertex> >(edm::InputTag("hiSelectedVertex"));

  // need to initialize the PF cand histograms : which are also event variables 
  if(isPFJet){
    mNPFpart = 0;
    mPFPt = 0;
    mPFEta = 0;
    mPFPhi = 0;
    mPFVsPt = 0;
    mPFVsPtInitial = 0;
    mPFVsPtInitial_HF = 0;
    mPFVsPtInitial_Barrel_Centrality_100_95 = 0;
    mPFVsPtInitial_Barrel_Centrality_95_90 = 0;
    mPFVsPtInitial_Barrel_Centrality_90_85 = 0;
    mPFVsPtInitial_Barrel_Centrality_85_80 = 0;
    mPFVsPtInitial_Barrel_Centrality_80_75 = 0;
    mPFVsPtInitial_Barrel_Centrality_75_70 = 0;
    mPFVsPtInitial_Barrel_Centrality_70_65 = 0;
    mPFVsPtInitial_Barrel_Centrality_65_60 = 0;
    mPFVsPtInitial_Barrel_Centrality_60_55 = 0;
    mPFVsPtInitial_Barrel_Centrality_55_50 = 0;
    mPFVsPtInitial_Barrel_Centrality_50_45 = 0;
    mPFVsPtInitial_Barrel_Centrality_45_40 = 0;
    mPFVsPtInitial_Barrel_Centrality_40_35 = 0;
    mPFVsPtInitial_Barrel_Centrality_35_30 = 0;
    mPFVsPtInitial_Barrel_Centrality_30_25 = 0;
    mPFVsPtInitial_Barrel_Centrality_25_20 = 0;
    mPFVsPtInitial_Barrel_Centrality_20_15 = 0;
    mPFVsPtInitial_Barrel_Centrality_15_10 = 0;
    mPFVsPtInitial_Barrel_Centrality_10_5 = 0;
    mPFVsPtInitial_Barrel_Centrality_5_0 = 0;

    mPFVsPtInitial_EndCap_Centrality_100_95 = 0;
    mPFVsPtInitial_EndCap_Centrality_95_90 = 0;
    mPFVsPtInitial_EndCap_Centrality_90_85 = 0;
    mPFVsPtInitial_EndCap_Centrality_85_80 = 0;
    mPFVsPtInitial_EndCap_Centrality_80_75 = 0;
    mPFVsPtInitial_EndCap_Centrality_75_70 = 0;
    mPFVsPtInitial_EndCap_Centrality_70_65 = 0;
    mPFVsPtInitial_EndCap_Centrality_65_60 = 0;
    mPFVsPtInitial_EndCap_Centrality_60_55 = 0;
    mPFVsPtInitial_EndCap_Centrality_55_50 = 0;
    mPFVsPtInitial_EndCap_Centrality_50_45 = 0;
    mPFVsPtInitial_EndCap_Centrality_45_40 = 0;
    mPFVsPtInitial_EndCap_Centrality_40_35 = 0;
    mPFVsPtInitial_EndCap_Centrality_35_30 = 0;
    mPFVsPtInitial_EndCap_Centrality_30_25 = 0;
    mPFVsPtInitial_EndCap_Centrality_25_20 = 0;
    mPFVsPtInitial_EndCap_Centrality_20_15 = 0;
    mPFVsPtInitial_EndCap_Centrality_15_10 = 0;
    mPFVsPtInitial_EndCap_Centrality_10_5 = 0;
    mPFVsPtInitial_EndCap_Centrality_5_0 = 0;

    mPFVsPtInitial_HF_Centrality_100_95 = 0;
    mPFVsPtInitial_HF_Centrality_95_90 = 0;
    mPFVsPtInitial_HF_Centrality_90_85 = 0;
    mPFVsPtInitial_HF_Centrality_85_80 = 0;
    mPFVsPtInitial_HF_Centrality_80_75 = 0;
    mPFVsPtInitial_HF_Centrality_75_70 = 0;
    mPFVsPtInitial_HF_Centrality_70_65 = 0;
    mPFVsPtInitial_HF_Centrality_65_60 = 0;
    mPFVsPtInitial_HF_Centrality_60_55 = 0;
    mPFVsPtInitial_HF_Centrality_55_50 = 0;
    mPFVsPtInitial_HF_Centrality_50_45 = 0;
    mPFVsPtInitial_HF_Centrality_45_40 = 0;
    mPFVsPtInitial_HF_Centrality_40_35 = 0;
    mPFVsPtInitial_HF_Centrality_35_30 = 0;
    mPFVsPtInitial_HF_Centrality_30_25 = 0;
    mPFVsPtInitial_HF_Centrality_25_20 = 0;
    mPFVsPtInitial_HF_Centrality_20_15 = 0;
    mPFVsPtInitial_HF_Centrality_15_10 = 0;
    mPFVsPtInitial_HF_Centrality_10_5 = 0;
    mPFVsPtInitial_HF_Centrality_5_0 = 0;

    mPFArea = 0;
    mPFDeltaR = 0; 
    mPFDeltaR_Scaled_R = 0; 

    mPFDeltaR_pTCorrected = 0; 
    mPFDeltaR_pTCorrected_PFpT_20To30 = 0; 
    mPFDeltaR_pTCorrected_PFpT_30To50 = 0; 
    mPFDeltaR_pTCorrected_PFpT_50To80 = 0; 
    mPFDeltaR_pTCorrected_PFpT_80To120 = 0; 
    mPFDeltaR_pTCorrected_PFpT_120To180 = 0; 
    mPFDeltaR_pTCorrected_PFpT_180To300 = 0; 
    mPFDeltaR_pTCorrected_PFpT_300ToInf = 0;  

    mPFDeltaR_pTCorrected_PFVsInitialpT_20To30 = 0;  
    mPFDeltaR_pTCorrected_PFVsInitialpT_30To50 = 0;  
    mPFDeltaR_pTCorrected_PFVsInitialpT_50To80 = 0;  
    mPFDeltaR_pTCorrected_PFVsInitialpT_80To120 = 0; 
    mPFDeltaR_pTCorrected_PFVsInitialpT_120To180 = 0;
    mPFDeltaR_pTCorrected_PFVsInitialpT_180To300 = 0;
    mPFDeltaR_pTCorrected_PFVsInitialpT_300ToInf = 0;


    mPFVsPtInitialDeltaR_pTCorrected = 0; 
    mPFVsPtDeltaR_pTCorrected = 0; 

    mSumPFVsPt = 0;
    mSumPFVsPtInitial = 0;
    mSumPFPt = 0;
    mSumPFVsPtInitial_eta = 0;
    mSumPFVsPt_eta = 0;
    mSumPFPt_eta = 0;
    mSumSquaredPFVsPt = 0;
    mSumSquaredPFVsPtInitial = 0;
    mSumSquaredPFPt = 0;
    mSumSquaredPFVsPtInitial_eta = 0;
    mSumSquaredPFVsPt_eta = 0;
    mSumSquaredPFPt_eta = 0;
    mSumPFVsPtInitial_HF = 0;
    mSumPFVsPt_HF = 0;
    mSumPFPt_HF = 0;

    mSumPFVsPtInitial_n5p191_n2p650 = 0;
    mSumPFVsPtInitial_n2p650_n2p043 = 0;
    mSumPFVsPtInitial_n2p043_n1p740 = 0;
    mSumPFVsPtInitial_n1p740_n1p479 = 0;
    mSumPFVsPtInitial_n1p479_n1p131 = 0;
    mSumPFVsPtInitial_n1p131_n0p783 = 0;
    mSumPFVsPtInitial_n0p783_n0p522 = 0;
    mSumPFVsPtInitial_n0p522_0p522 = 0;
    mSumPFVsPtInitial_0p522_0p783 = 0;
    mSumPFVsPtInitial_0p783_1p131 = 0;
    mSumPFVsPtInitial_1p131_1p479 = 0;
    mSumPFVsPtInitial_1p479_1p740 = 0;
    mSumPFVsPtInitial_1p740_2p043 = 0;
    mSumPFVsPtInitial_2p043_2p650 = 0;
    mSumPFVsPtInitial_2p650_5p191 = 0;

    mSumPFVsPt_n5p191_n2p650 = 0;
    mSumPFVsPt_n2p650_n2p043 = 0;
    mSumPFVsPt_n2p043_n1p740 = 0;
    mSumPFVsPt_n1p740_n1p479 = 0;
    mSumPFVsPt_n1p479_n1p131 = 0;
    mSumPFVsPt_n1p131_n0p783 = 0;
    mSumPFVsPt_n0p783_n0p522 = 0;
    mSumPFVsPt_n0p522_0p522 = 0;
    mSumPFVsPt_0p522_0p783 = 0;
    mSumPFVsPt_0p783_1p131 = 0;
    mSumPFVsPt_1p131_1p479 = 0;
    mSumPFVsPt_1p479_1p740 = 0;
    mSumPFVsPt_1p740_2p043 = 0;
    mSumPFVsPt_2p043_2p650 = 0;
    mSumPFVsPt_2p650_5p191 = 0;

    mSumPFPt_n5p191_n2p650 = 0;
    mSumPFPt_n2p650_n2p043 = 0;
    mSumPFPt_n2p043_n1p740 = 0;
    mSumPFPt_n1p740_n1p479 = 0;
    mSumPFPt_n1p479_n1p131 = 0;
    mSumPFPt_n1p131_n0p783 = 0;
    mSumPFPt_n0p783_n0p522 = 0;
    mSumPFPt_n0p522_0p522 = 0;
    mSumPFPt_0p522_0p783 = 0;
    mSumPFPt_0p783_1p131 = 0;
    mSumPFPt_1p131_1p479 = 0;
    mSumPFPt_1p479_1p740 = 0;
    mSumPFPt_1p740_2p043 = 0;
    mSumPFPt_2p043_2p650 = 0;
    mSumPFPt_2p650_5p191 = 0;

    mPFCandpT_vs_eta_Unknown = 0; // pf id 0
    mPFCandpT_vs_eta_ChargedHadron = 0; // pf id - 1 
    mPFCandpT_vs_eta_electron = 0; // pf id - 2
    mPFCandpT_vs_eta_muon = 0; // pf id - 3
    mPFCandpT_vs_eta_photon = 0; // pf id - 4
    mPFCandpT_vs_eta_NeutralHadron = 0; // pf id - 5
    mPFCandpT_vs_eta_HadE_inHF = 0; // pf id - 6
    mPFCandpT_vs_eta_EME_inHF = 0; // pf id - 7
     
    mPFCandpT_Barrel_Unknown = 0; // pf id 0
    mPFCandpT_Barrel_ChargedHadron = 0; // pf id - 1 
    mPFCandpT_Barrel_electron = 0; // pf id - 2
    mPFCandpT_Barrel_muon = 0; // pf id - 3
    mPFCandpT_Barrel_photon = 0; // pf id - 4
    mPFCandpT_Barrel_NeutralHadron = 0; // pf id - 5
    mPFCandpT_Barrel_HadE_inHF = 0; // pf id - 6
    mPFCandpT_Barrel_EME_inHF = 0; // pf id - 7

    mPFCandpT_Endcap_Unknown = 0; // pf id 0
    mPFCandpT_Endcap_ChargedHadron = 0; // pf id - 1 
    mPFCandpT_Endcap_electron = 0; // pf id - 2
    mPFCandpT_Endcap_muon = 0; // pf id - 3
    mPFCandpT_Endcap_photon = 0; // pf id - 4
    mPFCandpT_Endcap_NeutralHadron = 0; // pf id - 5
    mPFCandpT_Endcap_HadE_inHF = 0; // pf id - 6
    mPFCandpT_Endcap_EME_inHF = 0; // pf id - 7

    mPFCandpT_Forward_Unknown = 0; // pf id 0
    mPFCandpT_Forward_ChargedHadron = 0; // pf id - 1 
    mPFCandpT_Forward_electron = 0; // pf id - 2
    mPFCandpT_Forward_muon = 0; // pf id - 3
    mPFCandpT_Forward_photon = 0; // pf id - 4
    mPFCandpT_Forward_NeutralHadron = 0; // pf id - 5
    mPFCandpT_Forward_HadE_inHF = 0; // pf id - 6
    mPFCandpT_Forward_EME_inHF = 0; // pf id - 7
      
  }
  if(isCaloJet){
    mNCalopart = 0;
    mCaloPt = 0;
    mCaloEta = 0;
    mCaloPhi = 0;
    mCaloVsPt = 0;
    mCaloVsPtInitial = 0;
    mCaloArea = 0;

    mSumCaloVsPt = 0;
    mSumCaloVsPtInitial = 0;
    mSumCaloPt = 0;
    mSumCaloVsPtInitial_eta = 0;
    mSumCaloVsPt_eta = 0;
    mSumCaloPt_eta = 0;
    mSumSquaredCaloVsPt = 0;
    mSumSquaredCaloVsPtInitial = 0;
    mSumSquaredCaloPt = 0;
    mSumSquaredCaloVsPtInitial_eta = 0;
    mSumSquaredCaloVsPt_eta = 0;
    mSumSquaredCaloPt_eta = 0;
    mSumCaloVsPtInitial_HF = 0;
    mSumCaloVsPt_HF = 0;
    mSumCaloPt_HF = 0;

    mSumCaloVsPtInitial_n5p191_n2p650 = 0;
    mSumCaloVsPtInitial_n2p650_n2p043 = 0;
    mSumCaloVsPtInitial_n2p043_n1p740 = 0;
    mSumCaloVsPtInitial_n1p740_n1p479 = 0;
    mSumCaloVsPtInitial_n1p479_n1p131 = 0;
    mSumCaloVsPtInitial_n1p131_n0p783 = 0;
    mSumCaloVsPtInitial_n0p783_n0p522 = 0;
    mSumCaloVsPtInitial_n0p522_0p522 = 0;
    mSumCaloVsPtInitial_0p522_0p783 = 0;
    mSumCaloVsPtInitial_0p783_1p131 = 0;
    mSumCaloVsPtInitial_1p131_1p479 = 0;
    mSumCaloVsPtInitial_1p479_1p740 = 0;
    mSumCaloVsPtInitial_1p740_2p043 = 0;
    mSumCaloVsPtInitial_2p043_2p650 = 0;
    mSumCaloVsPtInitial_2p650_5p191 = 0;

    mSumCaloVsPt_n5p191_n2p650 = 0;
    mSumCaloVsPt_n2p650_n2p043 = 0;
    mSumCaloVsPt_n2p043_n1p740 = 0;
    mSumCaloVsPt_n1p740_n1p479 = 0;
    mSumCaloVsPt_n1p479_n1p131 = 0;
    mSumCaloVsPt_n1p131_n0p783 = 0;
    mSumCaloVsPt_n0p783_n0p522 = 0;
    mSumCaloVsPt_n0p522_0p522 = 0;
    mSumCaloVsPt_0p522_0p783 = 0;
    mSumCaloVsPt_0p783_1p131 = 0;
    mSumCaloVsPt_1p131_1p479 = 0;
    mSumCaloVsPt_1p479_1p740 = 0;
    mSumCaloVsPt_1p740_2p043 = 0;
    mSumCaloVsPt_2p043_2p650 = 0;
    mSumCaloVsPt_2p650_5p191 = 0;

    mSumCaloPt_n5p191_n2p650 = 0;
    mSumCaloPt_n2p650_n2p043 = 0;
    mSumCaloPt_n2p043_n1p740 = 0;
    mSumCaloPt_n1p740_n1p479 = 0;
    mSumCaloPt_n1p479_n1p131 = 0;
    mSumCaloPt_n1p131_n0p783 = 0;
    mSumCaloPt_n0p783_n0p522 = 0;
    mSumCaloPt_n0p522_0p522 = 0;
    mSumCaloPt_0p522_0p783 = 0;
    mSumCaloPt_0p783_1p131 = 0;
    mSumCaloPt_1p131_1p479 = 0;
    mSumCaloPt_1p479_1p740 = 0;
    mSumCaloPt_1p740_2p043 = 0;
    mSumCaloPt_2p043_2p650 = 0;
    mSumCaloPt_2p650_5p191 = 0;

  }

  mSumpt = 0;

  // Events variables
  mNvtx         = 0;
  mHF           = 0;

  // added Jan 12th 2015

  // Jet parameters
  mEta          = 0;
  mPhi          = 0;
  mEnergy       = 0;
  mP            = 0;
  mPt           = 0;
  mMass         = 0;
  mConstituents = 0;
  mJetArea      = 0;
  mjetpileup    = 0;
  mNJets_40     = 0;
  mNJets        = 0;


}

void JetAnalyzer_HeavyIons::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &) 
{

  ibooker.setCurrentFolder("JetMET/HIJetValidation/"+mInputCollection.label());

  double edge_pseudorapidity[etaBins_ +1] = {-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191 };

  TH2F *h2D_etabins_vs_pt2 = new TH2F("h2D_etabins_vs_pt2",";#eta;sum p_{T}^{2}",etaBins_,edge_pseudorapidity,10000,0,10000);
  TH2F *h2D_etabins_vs_pt = new TH2F("h2D_etabins_vs_pt",";#eta;sum p_{T}",etaBins_,edge_pseudorapidity,10000,-1000,1000);
  TH2F *h2D_pfcand_etabins_vs_pt = new TH2F("h2D_etabins_vs_pt",";#eta;sum p_{T}",etaBins_,edge_pseudorapidity,300, 0, 300);

  if(isPFJet){

    mNPFpart         = ibooker.book1D("NPFpart","No of particle flow candidates",1000,0,1000);
    mPFPt            = ibooker.book1D("PFPt","PF candidate p_{T}",10000,-500,500);
    mPFEta           = ibooker.book1D("PFEta","PF candidate #eta",120,-6,6);
    mPFPhi           = ibooker.book1D("PFPhi","PF candidate #phi",70,-3.5,3.5);
    mPFVsPt          = ibooker.book1D("PFVsPt","Vs PF candidate p_{T}",1000,-500,500);
    mPFVsPtInitial   = ibooker.book1D("PFVsPtInitial","Vs background subtracted PF candidate p_{T}",1000,-500,500);
    mPFVsPtInitial_HF   = ibooker.book2D("PFVsPtInitial_HF","Vs background subtracted PF candidate p_{T} versus HF", 600,0,6000,1000,-500,500);

    mPFVsPtInitial_Barrel_Centrality_100_95   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_100_95","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW100And95 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_95_90   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_95_90","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW95And90 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_90_85   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_90_85","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW90And85 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_85_80   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_85_80","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW85And80 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_80_75   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_80_75","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW80And75 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_75_70   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_75_70","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW75And70 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_70_65   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_70_65","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW70And65 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_65_60   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_65_60","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW65And60 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_60_55   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_60_55","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW60And55 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_55_50   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_55_50","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW55And50 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_50_45   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_50_45","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW50And45 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_45_40   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_45_40","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW45And40 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_40_35   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_40_35","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW40And35 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_35_30   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_35_30","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW35And30 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_30_25   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_30_25","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW30And25 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_25_20   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_25_20","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW25And20 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_20_15   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_20_15","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW20And15 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_15_10   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_15_10","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW15And10 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_10_5   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_10_5","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW10And05 Barrel",1000,-100,100);
    mPFVsPtInitial_Barrel_Centrality_5_0   = ibooker.book1D("mPFVsPtInitial_Barrel_Centrality_5_0","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW05And00 Barrel",1000,-100,100);

    mPFVsPtInitial_EndCap_Centrality_100_95   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_100_95","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW100And95 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_95_90   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_95_90","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW95And90 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_90_85   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_90_85","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW90And85 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_85_80   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_85_80","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW85And80 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_80_75   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_80_75","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW80And75 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_75_70   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_75_70","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW75And70 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_70_65   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_70_65","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW70And65 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_65_60   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_65_60","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW65And60 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_60_55   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_60_55","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW60And55 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_55_50   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_55_50","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW55And50 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_50_45   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_50_45","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW50And45 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_45_40   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_45_40","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW45And40 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_40_35   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_40_35","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW40And35 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_35_30   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_35_30","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW35And30 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_30_25   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_30_25","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW30And25 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_25_20   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_25_20","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW25And20 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_20_15   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_20_15","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW20And15 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_15_10   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_15_10","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW15And10 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_10_5   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_10_5","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW10And05 EndCap",1000,-100,100);
    mPFVsPtInitial_EndCap_Centrality_5_0   = ibooker.book1D("mPFVsPtInitial_EndCap_Centrality_5_0","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW05And00 EndCap",1000,-100,100);

    mPFVsPtInitial_HF_Centrality_100_95   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_100_95","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW100And95 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_95_90   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_95_90","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW95And90 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_90_85   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_90_85","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW90And85 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_85_80   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_85_80","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW85And80 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_80_75   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_80_75","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW80And75 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_75_70   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_75_70","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW75And70 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_70_65   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_70_65","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW70And65 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_65_60   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_65_60","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW65And60 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_60_55   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_60_55","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW60And55 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_55_50   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_55_50","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW55And50 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_50_45   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_50_45","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW50And45 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_45_40   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_45_40","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW45And40 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_40_35   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_40_35","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW40And35 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_35_30   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_35_30","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW35And30 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_30_25   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_30_25","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW30And25 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_25_20   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_25_20","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW25And20 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_20_15   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_20_15","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW20And15 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_15_10   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_15_10","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW15And10 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_10_5   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_10_5","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW10And05 HF",1000,-100,100);
    mPFVsPtInitial_HF_Centrality_5_0   = ibooker.book1D("mPFVsPtInitial_HF_Centrality_5_0","Vs background subtracted PF candidate p_{T} versus HF Energy Dist. BTW05And00 HF",1000,-100,100);

    mPFArea          = ibooker.book1D("PFArea","VS PF candidate area",100,0,4);
    mPFDeltaR          = ibooker.book1D("PFDeltaR","VS PF candidate DeltaR",100,0,4); //MZ
    mPFDeltaR_Scaled_R          = ibooker.book1D("PFDeltaR_Scaled_R","VS PF candidate DeltaR Divided by DeltaR square",100,0,4); //MZ
    mPFDeltaR_pTCorrected          = ibooker.book1D("PFDeltaR_pTCorrected","VS PF candidate DeltaR time pFpT over jet pT",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_20To30          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_20_30","VS PF candidate DeltaR time pFpT over jet pT for PFpT 20 to 30 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_30To50          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_30_50","VS PF candidate DeltaR time pFpT over jet pT for PFpT 30 to 50 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_50To80          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_50_80","VS PF candidate DeltaR time pFpT over jet pT for PFpT 50 to 80 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_80To120          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_80_120","VS PF candidate DeltaR time pFpT over jet pT for PFpT 80 to 120 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_120To180          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_120_180","VS PF candidate DeltaR time pFpT over jet pT for PFpT 120 to 180 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_180To300          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_180_300","VS PF candidate DeltaR time pFpT over jet pT for PFpT 180 to 300 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFpT_300ToInf          = ibooker.book1D("PFDeltaR_pTCorrected_PFpT_300_Inf","VS PF candidate DeltaR time pFpT over jet pT for PFpT 300 to Inf GeV",100,0,4); //MZ

    mPFDeltaR_pTCorrected_PFVsInitialpT_20To30          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_20_30","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 20 to 30 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFVsInitialpT_30To50          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_30_50","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 30 to 50 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFVsInitialpT_50To80          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_50_80","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 50 to 80 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFVsInitialpT_80To120          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_80_120","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 80 to 120 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFVsInitialpT_120To180          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_120_180","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 120 to 180 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFVsInitialpT_180To300          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_180_300","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 180 to 300 GeV",100,0,4); //MZ
    mPFDeltaR_pTCorrected_PFVsInitialpT_300ToInf          = ibooker.book1D("PFVsInitialDeltaR_pTCorrected_PFpT_300_Inf","VS PF candidate DeltaR time pFVsInitiaal pT over jet pT for PFpT 300 to Inf GeV",100,0,4); //MZ


    mPFVsPtInitialDeltaR_pTCorrected         = ibooker.book1D("PFVsPtInitialDeltaR_pTCorrected","VS PF Initial candidate DeltaR time pFpTInitial over jet pT",100,0,4); //MZ
    mPFVsPtDeltaR_pTCorrected         = ibooker.book1D("PFVsPtDeltaR_pTCorrected","VS PF candidate DeltaR time pFpT over jet pT",100,0,4); //MZ

    mSumPFVsPt       = ibooker.book1D("SumPFVsPt","Sum of final PF VS p_{T}",1000,-10000,10000);
    mSumPFVsPtInitial= ibooker.book1D("SumPFVsPtInitial","Sum PF VS p_{T} after subtraction",1000,-10000,10000);
    mSumPFPt         = ibooker.book1D("SumPFPt","Sum of initial PF p_{T}",1000,-10000,10000);
    mSumPFVsPt_eta   = ibooker.book2D("SumPFVsPt_etaBins",h2D_etabins_vs_pt);
    mSumPFVsPtInitial_eta   = ibooker.book2D("SumPFVsPtInitial_etaBins",h2D_etabins_vs_pt);
    mSumPFPt_eta     = ibooker.book2D("SumPFPt_etaBins",h2D_etabins_vs_pt);

    mSumSquaredPFVsPt       = ibooker.book1D("SumSquaredPFVsPt","Sum PF Vs p_{T} square",10000,0,10000);
    mSumSquaredPFVsPtInitial= ibooker.book1D("SumSquaredPFVsPtInitial","Sum PF Vs p_{T} square after subtraction ",10000,0,10000);
    mSumSquaredPFPt         = ibooker.book1D("SumSquaredPFPt","Sum of initial PF p_{T} squared",10000,0,10000);
    mSumSquaredPFVsPt_eta   = ibooker.book2D("SumSquaredPFVsPt_etaBins",h2D_etabins_vs_pt2);
    mSumSquaredPFVsPtInitial_eta   = ibooker.book2D("SumSquaredPFVsPtInitial_etaBins",h2D_etabins_vs_pt2);
    mSumSquaredPFPt_eta     = ibooker.book2D("SumSquaredPFPt_etaBins",h2D_etabins_vs_pt2);

    mSumPFVsPtInitial_HF    = ibooker.book2D("SumPFVsPtInitial_HF","HF Energy (y axis) vs Sum PF Vs p_{T} before subtraction (x axis)",1000,-1000,1000,1000,0,10000);
    mSumPFVsPt_HF    = ibooker.book2D("SumPFVsPt_HF","HF energy (y axis) vs Sum PF Vs p_{T} final (x axis)",1000,-1000,1000,1000,0,10000);
    mSumPFPt_HF    = ibooker.book2D("SumPFPt_HF","HF energy (y axis) vs Sum initial PF p_{T} (x axis)",1000,-1000,1000,1000,0,10000);

    mSumPFVsPtInitial_n5p191_n2p650 = ibooker.book1D("mSumPFVsPtInitial_n5p191_n2p650","Sum PFVsPt Initial variable in the eta range -5.191 to -2.650",1000,-5000,5000);
    mSumPFVsPtInitial_n2p650_n2p043 = ibooker.book1D("mSumPFVsPtInitial_n2p650_n2p043","Sum PFVsPt Initial variable in the eta range -2.650 to -2.043 ",1000,-5000,5000);
    mSumPFVsPtInitial_n2p043_n1p740 = ibooker.book1D("mSumPFVsPtInitial_n2p043_n1p740","Sum PFVsPt Initial variable in the eta range -2.043 to -1.740",1000,-1000,1000);
    mSumPFVsPtInitial_n1p740_n1p479 = ibooker.book1D("mSumPFVsPtInitial_n1p740_n1p479","Sum PFVsPt Initial variable in the eta range -1.740 to -1.479",1000,-1000,1000);
    mSumPFVsPtInitial_n1p479_n1p131 = ibooker.book1D("mSumPFVsPtInitial_n1p479_n1p131","Sum PFVsPt Initial variable in the eta range -1.479 to -1.131",1000,-1000,1000);
    mSumPFVsPtInitial_n1p131_n0p783 = ibooker.book1D("mSumPFVsPtInitial_n1p131_n0p783","Sum PFVsPt Initial variable in the eta range -1.131 to -0.783",1000,-1000,1000);
    mSumPFVsPtInitial_n0p783_n0p522 = ibooker.book1D("mSumPFVsPtInitial_n0p783_n0p522","Sum PFVsPt Initial variable in the eta range -0.783 to -0.522",1000,-1000,1000);
    mSumPFVsPtInitial_n0p522_0p522 = ibooker.book1D("mSumPFVsPtInitial_n0p522_0p522","Sum PFVsPt Initial variable in the eta range -0.522 to 0.522",1000,-1000,1000);
    mSumPFVsPtInitial_0p522_0p783 = ibooker.book1D("mSumPFVsPtInitial_0p522_0p783","Sum PFVsPt Initial variable in the eta range 0.522 to 0.783",1000,-1000,1000);
    mSumPFVsPtInitial_0p783_1p131 = ibooker.book1D("mSumPFVsPtInitial_0p783_1p131","Sum PFVsPt Initial variable in the eta range 0.783 to 1.131",1000,-1000,1000);
    mSumPFVsPtInitial_1p131_1p479 = ibooker.book1D("mSumPFVsPtInitial_1p131_1p479","Sum PFVsPt Initial variable in the eta range 1.131 to 1.479",1000,-1000,1000);
    mSumPFVsPtInitial_1p479_1p740 = ibooker.book1D("mSumPFVsPtInitial_1p479_1p740","Sum PFVsPt Initial variable in the eta range 1.479 to 1.740",1000,-1000,1000);
    mSumPFVsPtInitial_1p740_2p043 = ibooker.book1D("mSumPFVsPtInitial_1p740_2p043","Sum PFVsPt Initial variable in the eta range 1.740 to 2.043",1000,-1000,1000);
    mSumPFVsPtInitial_2p043_2p650 = ibooker.book1D("mSumPFVsPtInitial_2p043_2p650","Sum PFVsPt Initial variable in the eta range 2.043 to 2.650",1000,-5000,5000);
    mSumPFVsPtInitial_2p650_5p191 = ibooker.book1D("mSumPFVsPtInitial_2p650_5p191","Sum PFVsPt Initial variable in the eta range 2.650 to 5.191",1000,-5000,5000);

    mSumPFVsPt_n5p191_n2p650 = ibooker.book1D("mSumPFVsPt_n5p191_n2p650","Sum PFVsPt  variable in the eta range -5.191 to -2.650",1000,-5000,5000);
    mSumPFVsPt_n2p650_n2p043 = ibooker.book1D("mSumPFVsPt_n2p650_n2p043","Sum PFVsPt  variable in the eta range -2.650 to -2.043 ",1000,-5000,5000);
    mSumPFVsPt_n2p043_n1p740 = ibooker.book1D("mSumPFVsPt_n2p043_n1p740","Sum PFVsPt  variable in the eta range -2.043 to -1.740",1000,-1000,1000);
    mSumPFVsPt_n1p740_n1p479 = ibooker.book1D("mSumPFVsPt_n1p740_n1p479","Sum PFVsPt  variable in the eta range -1.740 to -1.479",1000,-1000,1000);
    mSumPFVsPt_n1p479_n1p131 = ibooker.book1D("mSumPFVsPt_n1p479_n1p131","Sum PFVsPt  variable in the eta range -1.479 to -1.131",1000,-1000,1000);
    mSumPFVsPt_n1p131_n0p783 = ibooker.book1D("mSumPFVsPt_n1p131_n0p783","Sum PFVsPt  variable in the eta range -1.131 to -0.783",1000,-1000,1000);
    mSumPFVsPt_n0p783_n0p522 = ibooker.book1D("mSumPFVsPt_n0p783_n0p522","Sum PFVsPt  variable in the eta range -0.783 to -0.522",1000,-1000,1000);
    mSumPFVsPt_n0p522_0p522 = ibooker.book1D("mSumPFVsPt_n0p522_0p522","Sum PFVsPt  variable in the eta range -0.522 to 0.522",1000,-1000,1000);
    mSumPFVsPt_0p522_0p783 = ibooker.book1D("mSumPFVsPt_0p522_0p783","Sum PFVsPt  variable in the eta range 0.522 to 0.783",1000,-1000,1000);
    mSumPFVsPt_0p783_1p131 = ibooker.book1D("mSumPFVsPt_0p783_1p131","Sum PFVsPt  variable in the eta range 0.783 to 1.131",1000,-1000,1000);
    mSumPFVsPt_1p131_1p479 = ibooker.book1D("mSumPFVsPt_1p131_1p479","Sum PFVsPt  variable in the eta range 1.131 to 1.479",1000,-1000,1000);
    mSumPFVsPt_1p479_1p740 = ibooker.book1D("mSumPFVsPt_1p479_1p740","Sum PFVsPt  variable in the eta range 1.479 to 1.740",1000,-1000,1000);
    mSumPFVsPt_1p740_2p043 = ibooker.book1D("mSumPFVsPt_1p740_2p043","Sum PFVsPt  variable in the eta range 1.740 to 2.043",1000,-1000,1000);
    mSumPFVsPt_2p043_2p650 = ibooker.book1D("mSumPFVsPt_2p043_2p650","Sum PFVsPt  variable in the eta range 2.043 to 2.650",1000,-5000,5000);
    mSumPFVsPt_2p650_5p191 = ibooker.book1D("mSumPFVsPt_2p650_5p191","Sum PFVsPt  variable in the eta range 2.650 to 5.191",1000,-5000,5000);

    mSumPFPt_n5p191_n2p650 = ibooker.book1D("mSumPFPt_n5p191_n2p650","Sum PFPt  in the eta range -5.191 to -2.650",1000,-5000,5000);
    mSumPFPt_n2p650_n2p043 = ibooker.book1D("mSumPFPt_n2p650_n2p043","Sum PFPt  in the eta range -2.650 to -2.043 ",1000,-5000,5000);
    mSumPFPt_n2p043_n1p740 = ibooker.book1D("mSumPFPt_n2p043_n1p740","Sum PFPt  in the eta range -2.043 to -1.740",1000,-1000,1000);
    mSumPFPt_n1p740_n1p479 = ibooker.book1D("mSumPFPt_n1p740_n1p479","Sum PFPt  in the eta range -1.740 to -1.479",1000,-1000,1000);
    mSumPFPt_n1p479_n1p131 = ibooker.book1D("mSumPFPt_n1p479_n1p131","Sum PFPt  in the eta range -1.479 to -1.131",1000,-1000,1000);
    mSumPFPt_n1p131_n0p783 = ibooker.book1D("mSumPFPt_n1p131_n0p783","Sum PFPt  in the eta range -1.131 to -0.783",1000,-1000,1000);
    mSumPFPt_n0p783_n0p522 = ibooker.book1D("mSumPFPt_n0p783_n0p522","Sum PFPt  in the eta range -0.783 to -0.522",1000,-1000,1000);
    mSumPFPt_n0p522_0p522 = ibooker.book1D("mSumPFPt_n0p522_0p522","Sum PFPt  in the eta range -0.522 to 0.522",1000,-1000,1000);
    mSumPFPt_0p522_0p783 = ibooker.book1D("mSumPFPt_0p522_0p783","Sum PFPt  in the eta range 0.522 to 0.783",1000,-1000,1000);
    mSumPFPt_0p783_1p131 = ibooker.book1D("mSumPFPt_0p783_1p131","Sum PFPt  in the eta range 0.783 to 1.131",1000,-1000,1000);
    mSumPFPt_1p131_1p479 = ibooker.book1D("mSumPFPt_1p131_1p479","Sum PFPt  in the eta range 1.131 to 1.479",1000,-1000,1000);
    mSumPFPt_1p479_1p740 = ibooker.book1D("mSumPFPt_1p479_1p740","Sum PFPt  in the eta range 1.479 to 1.740",1000,-1000,1000);
    mSumPFPt_1p740_2p043 = ibooker.book1D("mSumPFPt_1p740_2p043","Sum PFPt  in the eta range 1.740 to 2.043",1000,-1000,1000);
    mSumPFPt_2p043_2p650 = ibooker.book1D("mSumPFPt_2p043_2p650","Sum PFPt  in the eta range 2.043 to 2.650",1000,-5000,5000);
    mSumPFPt_2p650_5p191 = ibooker.book1D("mSumPFPt_2p650_5p191","Sum PFPt  in the eta range 2.650 to 5.191",1000,-5000,5000);

    mPFCandpT_vs_eta_Unknown = ibooker.book2D("PF_cand_X_unknown",h2D_pfcand_etabins_vs_pt); // pf id 0
    mPFCandpT_vs_eta_ChargedHadron = ibooker.book2D("PF_cand_chargedHad",h2D_pfcand_etabins_vs_pt); // pf id - 1 
    mPFCandpT_vs_eta_electron = ibooker.book2D("PF_cand_electron",h2D_pfcand_etabins_vs_pt); // pf id - 2
    mPFCandpT_vs_eta_muon = ibooker.book2D("PF_cand_muon",h2D_pfcand_etabins_vs_pt); // pf id - 3
    mPFCandpT_vs_eta_photon = ibooker.book2D("PF_cand_photon",h2D_pfcand_etabins_vs_pt); // pf id - 4
    mPFCandpT_vs_eta_NeutralHadron = ibooker.book2D("PF_cand_neutralHad",h2D_pfcand_etabins_vs_pt); // pf id - 5
    mPFCandpT_vs_eta_HadE_inHF = ibooker.book2D("PF_cand_HadEner_inHF",h2D_pfcand_etabins_vs_pt); // pf id - 6
    mPFCandpT_vs_eta_EME_inHF = ibooker.book2D("PF_cand_EMEner_inHF",h2D_pfcand_etabins_vs_pt); // pf id - 7

    mPFCandpT_Barrel_Unknown = ibooker.book1D("mPFCandpT_Barrel_Unknown",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id  - 0
    mPFCandpT_Barrel_ChargedHadron = ibooker.book1D("mPFCandpT_Barrel_ChargedHadron",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 1 
    mPFCandpT_Barrel_electron = ibooker.book1D("mPFCandpT_Barrel_electron",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 2
    mPFCandpT_Barrel_muon = ibooker.book1D("mPFCandpT_Barrel_muon",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 3
    mPFCandpT_Barrel_photon = ibooker.book1D("mPFCandpT_Barrel_photon",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 4
    mPFCandpT_Barrel_NeutralHadron = ibooker.book1D("mPFCandpT_Barrel_NeutralHadron",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 5
    mPFCandpT_Barrel_HadE_inHF = ibooker.book1D("mPFCandpT_Barrel_HadE_inHF",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 6
    mPFCandpT_Barrel_EME_inHF = ibooker.book1D("mPFCandpT_Barrel_EME_inHF",Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),300, 0, 300); // pf id - 7
    
    mPFCandpT_Endcap_Unknown = ibooker.book1D("mPFCandpT_Endcap_Unknown",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 0
    mPFCandpT_Endcap_ChargedHadron = ibooker.book1D("mPFCandpT_Endcap_ChargedHadron",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 1 
    mPFCandpT_Endcap_electron = ibooker.book1D("mPFCandpT_Endcap_electron",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 2
    mPFCandpT_Endcap_muon = ibooker.book1D("mPFCandpT_Endcap_muon",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 3
    mPFCandpT_Endcap_photon = ibooker.book1D("mPFCandpT_Endcap_photon",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 4
    mPFCandpT_Endcap_NeutralHadron = ibooker.book1D("mPFCandpT_Endcap_NeutralHadron",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 5
    mPFCandpT_Endcap_HadE_inHF = ibooker.book1D("mPFCandpT_Endcap_HadE_inHF",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 6
    mPFCandpT_Endcap_EME_inHF = ibooker.book1D("mPFCandpT_Endcap_EME_inHF",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),300, 0, 300); // pf id - 7

    mPFCandpT_Forward_Unknown = ibooker.book1D("mPFCandpT_Forward_Unknown",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 0
    mPFCandpT_Forward_ChargedHadron = ibooker.book1D("mPFCandpT_Forward_ChargedHadron",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 1 
    mPFCandpT_Forward_electron = ibooker.book1D("mPFCandpT_Forward_electron",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 2
    mPFCandpT_Forward_muon = ibooker.book1D("mPFCandpT_Forward_muon",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 3
    mPFCandpT_Forward_photon = ibooker.book1D("mPFCandpT_Forward_photon",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 4
    mPFCandpT_Forward_NeutralHadron = ibooker.book1D("mPFCandpT_Forward_NeutralHadron",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 5
    mPFCandpT_Forward_HadE_inHF = ibooker.book1D("mPFCandpT_Forward_HadE_inHF",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 6
    mPFCandpT_Forward_EME_inHF = ibooker.book1D("mPFCandpT_Forward_EME_inHF",Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),300, 0, 300); // pf id - 7
        
  }

  if(isCaloJet){

    mNCalopart         = ibooker.book1D("NCalopart","No of particle flow candidates",1000,0,10000);
    mCaloPt            = ibooker.book1D("CaloPt","Calo candidate p_{T}",1000,-5000,5000);
    mCaloEta           = ibooker.book1D("CaloEta","Calo candidate #eta",120,-6,6);
    mCaloPhi           = ibooker.book1D("CaloPhi","Calo candidate #phi",70,-3.5,3.5);
    mCaloVsPt          = ibooker.book1D("CaloVsPt","Vs Calo candidate p_{T}",1000,-5000,5000);
    mCaloVsPtInitial   = ibooker.book1D("CaloVsPtInitial","Vs background subtracted Calo candidate p_{T}",1000,-500,500);
    mCaloArea          = ibooker.book1D("CaloArea","VS Calo candidate area",100,0,4);

    mSumCaloVsPt       = ibooker.book1D("SumCaloVsPt","Sum of final Calo VS p_{T} ",1000,-10000,10000);
    mSumCaloVsPtInitial= ibooker.book1D("SumCaloVsPtInitial","Sum Calo VS p_{T} after subtraction",1000,-10000,10000);
    mSumCaloPt         = ibooker.book1D("SumCaloPt","Sum Calo p_{T}",1000,-10000,10000);
    mSumCaloVsPt_eta   = ibooker.book2D("SumCaloVsPt_etaBins",h2D_etabins_vs_pt);
    mSumCaloVsPtInitial_eta   = ibooker.book2D("SumCaloVsPtInitial_etaBins",h2D_etabins_vs_pt);
    mSumCaloPt_eta     = ibooker.book2D("SumCaloPt_etaBins",h2D_etabins_vs_pt);

    mSumSquaredCaloVsPt       = ibooker.book1D("SumSquaredCaloVsPt","Sum of final Calo VS p_{T} squared",10000,0,10000);
    mSumSquaredCaloVsPtInitial= ibooker.book1D("SumSquaredCaloVsPtInitial","Sum of subtracted Calo VS p_{T} squared",10000,0,10000);
    mSumSquaredCaloPt         = ibooker.book1D("SumSquaredCaloPt","Sum of initial Calo tower p_{T} squared",10000,0,10000);
    mSumSquaredCaloVsPt_eta   = ibooker.book2D("SumSquaredCaloVsPt_etaBins",h2D_etabins_vs_pt2);
    mSumSquaredCaloVsPtInitial_eta   = ibooker.book2D("SumSquaredCaloVsPtInitial_etaBins",h2D_etabins_vs_pt2);
    mSumSquaredCaloPt_eta     = ibooker.book2D("SumSquaredCaloPt_etaBins",h2D_etabins_vs_pt2);

    mSumCaloVsPtInitial_HF    = ibooker.book2D("SumCaloVsPtInitial_HF","HF Energy (y axis) vs Sum Calo Vs p_{T} before subtraction (x axis)",1000,-1000,1000,1000,0,10000);
    mSumCaloVsPt_HF    = ibooker.book2D("SumCaloVsPt_HF","HF Energy (y axis) vs Sum Calo Vs p_{T} (x axis)",1000,-1000,1000,1000,0,10000);
    mSumCaloPt_HF    = ibooker.book2D("SumCaloPt_HF","HF Energy (y axis) vs Sum Calo tower p_{T}",1000,-1000,1000,1000,0,10000);

    mSumCaloVsPtInitial_n5p191_n2p650 = ibooker.book1D("mSumCaloVsPtInitial_n5p191_n2p650","Sum CaloVsPt Initial variable in the eta range -5.191 to -2.650",1000,-5000,5000);
    mSumCaloVsPtInitial_n2p650_n2p043 = ibooker.book1D("mSumCaloVsPtInitial_n2p650_n2p043","Sum CaloVsPt Initial variable in the eta range -2.650 to -2.043 ",1000,-5000,5000);
    mSumCaloVsPtInitial_n2p043_n1p740 = ibooker.book1D("mSumCaloVsPtInitial_n2p043_n1p740","Sum CaloVsPt Initial variable in the eta range -2.043 to -1.740",1000,-1000,1000);
    mSumCaloVsPtInitial_n1p740_n1p479 = ibooker.book1D("mSumCaloVsPtInitial_n1p740_n1p479","Sum CaloVsPt Initial variable in the eta range -1.740 to -1.479",1000,-1000,1000);
    mSumCaloVsPtInitial_n1p479_n1p131 = ibooker.book1D("mSumCaloVsPtInitial_n1p479_n1p131","Sum CaloVsPt Initial variable in the eta range -1.479 to -1.131",1000,-1000,1000);
    mSumCaloVsPtInitial_n1p131_n0p783 = ibooker.book1D("mSumCaloVsPtInitial_n1p131_n0p783","Sum CaloVsPt Initial variable in the eta range -1.131 to -0.783",1000,-1000,1000);
    mSumCaloVsPtInitial_n0p783_n0p522 = ibooker.book1D("mSumCaloVsPtInitial_n0p783_n0p522","Sum CaloVsPt Initial variable in the eta range -0.783 to -0.522",1000,-1000,1000);
    mSumCaloVsPtInitial_n0p522_0p522 = ibooker.book1D("mSumCaloVsPtInitial_n0p522_0p522","Sum CaloVsPt Initial variable in the eta range -0.522 to 0.522",1000,-1000,1000);
    mSumCaloVsPtInitial_0p522_0p783 = ibooker.book1D("mSumCaloVsPtInitial_0p522_0p783","Sum CaloVsPt Initial variable in the eta range 0.522 to 0.783",1000,-1000,1000);
    mSumCaloVsPtInitial_0p783_1p131 = ibooker.book1D("mSumCaloVsPtInitial_0p783_1p131","Sum CaloVsPt Initial variable in the eta range 0.783 to 1.131",1000,-1000,1000);
    mSumCaloVsPtInitial_1p131_1p479 = ibooker.book1D("mSumCaloVsPtInitial_1p131_1p479","Sum CaloVsPt Initial variable in the eta range 1.131 to 1.479",1000,-1000,1000);
    mSumCaloVsPtInitial_1p479_1p740 = ibooker.book1D("mSumCaloVsPtInitial_1p479_1p740","Sum CaloVsPt Initial variable in the eta range 1.479 to 1.740",1000,-1000,1000);
    mSumCaloVsPtInitial_1p740_2p043 = ibooker.book1D("mSumCaloVsPtInitial_1p740_2p043","Sum CaloVsPt Initial variable in the eta range 1.740 to 2.043",1000,-1000,1000);
    mSumCaloVsPtInitial_2p043_2p650 = ibooker.book1D("mSumCaloVsPtInitial_2p043_2p650","Sum CaloVsPt Initial variable in the eta range 2.043 to 2.650",1000,-5000,5000);
    mSumCaloVsPtInitial_2p650_5p191 = ibooker.book1D("mSumCaloVsPtInitial_2p650_5p191","Sum CaloVsPt Initial variable in the eta range 2.650 to 5.191",1000,-5000,5000);

    mSumCaloVsPt_n5p191_n2p650 = ibooker.book1D("mSumCaloVsPt_n5p191_n2p650","Sum CaloVsPt  variable in the eta range -5.191 to -2.650",1000,-5000,5000);
    mSumCaloVsPt_n2p650_n2p043 = ibooker.book1D("mSumCaloVsPt_n2p650_n2p043","Sum CaloVsPt  variable in the eta range -2.650 to -2.043",1000,-5000,5000);
    mSumCaloVsPt_n2p043_n1p740 = ibooker.book1D("mSumCaloVsPt_n2p043_n1p740","Sum CaloVsPt  variable in the eta range -2.043 to -1.740",1000,-1000,1000);
    mSumCaloVsPt_n1p740_n1p479 = ibooker.book1D("mSumCaloVsPt_n1p740_n1p479","Sum CaloVsPt  variable in the eta range -1.740 to -1.479",1000,-1000,1000);
    mSumCaloVsPt_n1p479_n1p131 = ibooker.book1D("mSumCaloVsPt_n1p479_n1p131","Sum CaloVsPt  variable in the eta range -1.479 to -1.131",1000,-1000,1000);
    mSumCaloVsPt_n1p131_n0p783 = ibooker.book1D("mSumCaloVsPt_n1p131_n0p783","Sum CaloVsPt  variable in the eta range -1.131 to -0.783",1000,-1000,1000);
    mSumCaloVsPt_n0p783_n0p522 = ibooker.book1D("mSumCaloVsPt_n0p783_n0p522","Sum CaloVsPt  variable in the eta range -0.783 to -0.522",1000,-1000,1000);
    mSumCaloVsPt_n0p522_0p522 = ibooker.book1D("mSumCaloVsPt_n0p522_0p522","Sum CaloVsPt  variable in the eta range -0.522 to 0.522",1000,-1000,1000);
    mSumCaloVsPt_0p522_0p783 = ibooker.book1D("mSumCaloVsPt_0p522_0p783","Sum CaloVsPt  variable in the eta range 0.522 to 0.783",1000,-1000,1000);
    mSumCaloVsPt_0p783_1p131 = ibooker.book1D("mSumCaloVsPt_0p783_1p131","Sum CaloVsPt  variable in the eta range 0.783 to 1.131",1000,-1000,1000);
    mSumCaloVsPt_1p131_1p479 = ibooker.book1D("mSumCaloVsPt_1p131_1p479","Sum CaloVsPt  variable in the eta range 1.131 to 1.479",1000,-1000,1000);
    mSumCaloVsPt_1p479_1p740 = ibooker.book1D("mSumCaloVsPt_1p479_1p740","Sum CaloVsPt  variable in the eta range 1.479 to 1.740",1000,-1000,1000);
    mSumCaloVsPt_1p740_2p043 = ibooker.book1D("mSumCaloVsPt_1p740_2p043","Sum CaloVsPt  variable in the eta range 1.740 to 2.043",1000,-1000,1000);
    mSumCaloVsPt_2p043_2p650 = ibooker.book1D("mSumCaloVsPt_2p043_2p650","Sum CaloVsPt  variable in the eta range 2.043 to 2.650",1000,-5000,5000);
    mSumCaloVsPt_2p650_5p191 = ibooker.book1D("mSumCaloVsPt_2p650_5p191","Sum CaloVsPt  variable in the eta range 2.650 to 5.191",1000,-5000,5000);

    mSumCaloPt_n5p191_n2p650 = ibooker.book1D("mSumCaloPt_n5p191_n2p650","Sum Calo tower pT variable in the eta range -5.191 to -2.650",1000,-5000,5000);
    mSumCaloPt_n2p650_n2p043 = ibooker.book1D("mSumCaloPt_n2p650_n2p043","Sum Calo tower pT variable in the eta range -2.650 to -2.043",1000,-5000,5000);
    mSumCaloPt_n2p043_n1p740 = ibooker.book1D("mSumCaloPt_n2p043_n1p740","Sum Calo tower pT variable in the eta range -2.043 to -1.740",1000,-1000,1000);
    mSumCaloPt_n1p740_n1p479 = ibooker.book1D("mSumCaloPt_n1p740_n1p479","Sum Calo tower pT variable in the eta range -1.740 to -1.479",1000,-1000,1000);
    mSumCaloPt_n1p479_n1p131 = ibooker.book1D("mSumCaloPt_n1p479_n1p131","Sum Calo tower pT variable in the eta range -1.479 to -1.131",1000,-1000,1000);
    mSumCaloPt_n1p131_n0p783 = ibooker.book1D("mSumCaloPt_n1p131_n0p783","Sum Calo tower pT variable in the eta range -1.131 to -0.783",1000,-1000,1000);
    mSumCaloPt_n0p783_n0p522 = ibooker.book1D("mSumCaloPt_n0p783_n0p522","Sum Calo tower pT variable in the eta range -0.783 to -0.522",1000,-1000,1000);
    mSumCaloPt_n0p522_0p522 = ibooker.book1D("mSumCaloPt_n0p522_0p522","Sum Calo tower pT variable in the eta range -0.522 to 0.522",1000,-1000,1000);
    mSumCaloPt_0p522_0p783 = ibooker.book1D("mSumCaloPt_0p522_0p783","Sum Calo tower pT variable in the eta range 0.522 to 0.783",1000,-1000,1000);
    mSumCaloPt_0p783_1p131 = ibooker.book1D("mSumCaloPt_0p783_1p131","Sum Calo tower pT variable in the eta range 0.783 to 1.131",1000,-1000,1000);
    mSumCaloPt_1p131_1p479 = ibooker.book1D("mSumCaloPt_1p131_1p479","Sum Calo tower pT variable in the eta range 1.131 to 1.479",1000,-1000,1000);
    mSumCaloPt_1p479_1p740 = ibooker.book1D("mSumCaloPt_1p479_1p740","Sum Calo tower pT variable in the eta range 1.479 to 1.740",1000,-1000,1000);
    mSumCaloPt_1p740_2p043 = ibooker.book1D("mSumCaloPt_1p740_2p043","Sum Calo tower pT variable in the eta range 1.740 to 2.043",1000,-1000,1000);
    mSumCaloPt_2p043_2p650 = ibooker.book1D("mSumCaloPt_2p043_2p650","Sum Calo tower pT variable in the eta range 2.043 to 2.650",1000,-5000,5000);
    mSumCaloPt_2p650_5p191 = ibooker.book1D("mSumCaloPt_2p650_5p191","Sum Calo tower pT variable in the eta range 2.650 to 5.191",1000,-5000,5000);


  }

  // particle flow variables histograms 
  mSumpt           = ibooker.book1D("SumpT","Sum p_{T} of all the PF candidates per event",1000,0,10000);

  // Event variables
  mNvtx            = ibooker.book1D("Nvtx",           "number of vertices", 60, 0, 60);
  mHF              = ibooker.book1D("HF", "HF energy distribution",1000,0,10000);

  // Jet parameters
  mEta             = ibooker.book1D("Eta",          "Eta",          120,   -6,    6); 
  mPhi             = ibooker.book1D("Phi",          "Phi",           70, -3.5,  3.5); 
  mPt              = ibooker.book1D("Pt",           "Pt",           100,    0,  1000); 
  mP               = ibooker.book1D("P",            "P",            100,    0,  1000); 
  mEnergy          = ibooker.book1D("Energy",       "Energy",       100,    0,  1000); 
  mMass            = ibooker.book1D("Mass",         "Mass",         100,    0,  200); 
  mConstituents    = ibooker.book1D("Constituents", "Constituents", 100,    0,  100); 
  mJetArea         = ibooker.book1D("JetArea",      "JetArea",       100,   0, 4);
  mjetpileup       = ibooker.book1D("jetPileUp","jetPileUp",100,0,150);
  mNJets_40        = ibooker.book1D("NJets_pt_greater_40", "NJets pT > 40 GeV",  100,    0,   100);
  mNJets           = ibooker.book1D("NJets", "NJets",  100,    0,   50);

  if (mOutputFile.empty ()) 
    LogInfo("OutputInfo") << " Histograms will NOT be saved";
  else 
    LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;

  delete h2D_etabins_vs_pt2;
  delete h2D_etabins_vs_pt;
  delete h2D_pfcand_etabins_vs_pt;
  
}



//------------------------------------------------------------------------------
// ~JetAnalyzer_HeavyIons
//------------------------------------------------------------------------------
JetAnalyzer_HeavyIons::~JetAnalyzer_HeavyIons() {}


//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
//void JetAnalyzer_HeavyIons::beginJob() {
//  std:://cout<<"inside the begin job function"<<endl;
//}


//------------------------------------------------------------------------------
// endJob
//------------------------------------------------------------------------------
//void JetAnalyzer_HeavyIons::endJob()
//{
//  if (!mOutputFile.empty() && &*edm::Service<DQMStore>())
//    {
//      edm::Service<DQMStore>()->save(mOutputFile);
//    }
//}


//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetAnalyzer_HeavyIons::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{

  // switch(mEvent.id().event() == 15296770)
  // case 1:
  //   break;

  // Get the primary vertices
  edm::Handle<vector<reco::Vertex> > pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);
  reco::Vertex::Point vtx(0,0,0);
  edm::Handle<reco::VertexCollection> vtxs;

  mEvent.getByToken(hiVertexToken_, vtxs);
  int greatestvtx = 0;
  int nVertex = vtxs->size();

  for (unsigned int i = 0 ; i< vtxs->size(); ++i){
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if( daughter > (*vtxs)[greatestvtx].tracksSize()) greatestvtx = i;
  }

  if(nVertex<=0){
    vtx =  reco::Vertex::Point(0,0,0);
  }
  vtx = (*vtxs)[greatestvtx].position();

  int nGoodVertices = 0;

  if (pvHandle.isValid())
    {
      for (unsigned i=0; i<pvHandle->size(); i++)
	{
	  if ((*pvHandle)[i].ndof() > 4 &&
	      (fabs((*pvHandle)[i].z()) <= 24) &&
	      (fabs((*pvHandle)[i].position().rho()) <= 2))
	    nGoodVertices++;
	}
    }

  mNvtx->Fill(nGoodVertices);

  // Get the Jet collection
  //----------------------------------------------------------------------------

  std::vector<Jet> recoJets;
  recoJets.clear();

  edm::Handle<CaloJetCollection>  caloJets;
  edm::Handle<JPTJetCollection>   jptJets;
  edm::Handle<PFJetCollection>    pfJets;
  edm::Handle<BasicJetCollection> basicJets;

  // Get the Particle flow candidates and the Voronoi variables 
  edm::Handle<reco::PFCandidateCollection> pfCandidates; 
  edm::Handle<CaloTowerCollection> caloCandidates; 
  edm::Handle<reco::CandidateView> pfcandidates_;
  edm::Handle<reco::CandidateView> calocandidates_;

  edm::Handle<edm::ValueMap<VoronoiBackground>> VsBackgrounds;
  edm::Handle<std::vector<float>> vn_;

  if (isCaloJet) mEvent.getByToken(caloJetsToken_, caloJets);
  if (isJPTJet)  mEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet) {  
    if(std::string("Pu")==UEAlgo) mEvent.getByToken(basicJetsToken_, basicJets);
    if(std::string("Vs")==UEAlgo) mEvent.getByToken(pfJetsToken_, pfJets);
  }

  mEvent.getByToken(pfCandToken_, pfCandidates);
  mEvent.getByToken(pfCandViewToken_, pfcandidates_);

  mEvent.getByToken(caloTowersToken_, caloCandidates);
  mEvent.getByToken(caloCandViewToken_, calocandidates_);

  mEvent.getByToken(backgrounds_, VsBackgrounds);
  mEvent.getByToken(backgrounds_value_, vn_);

  // get the centrality 
  edm::Handle<reco::Centrality> cent;
  mEvent.getByToken(centralityToken, cent);  //_centralitytag comes from the cfg

  mHF->Fill(cent->EtHFtowerSum());
  Float_t HF_energy = cent->EtHFtowerSum();
  
  edm::Handle<int> cbin;
  mEvent.getByToken(centralityBinToken, cbin);
  
  if (!cent.isValid()) return;
  int hibin = -999;
  if(cent.isValid())
    hibin = *cbin;
  
  
  const reco::PFCandidateCollection *pfCandidateColl = pfCandidates.product();

  Float_t vsPt=0;
  Float_t vsPtInitial = 0;
  Float_t vsArea = 0;
  Int_t   NPFpart = 0;
  Int_t   NCaloTower = 0;
  Float_t pfPt = 0;
  Float_t pfEta = 0;
  Float_t pfPhi = 0;
  Int_t   pfID = 0;
  Float_t pfDeltaR = 0; 
  Float_t caloPt = 0;
  Float_t caloEta = 0;
  Float_t caloPhi = 0;
  Float_t SumPt_value = 0;

  double edge_pseudorapidity[etaBins_ +1] = {-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191 };


  vector < vector <float> > numbers;
  vector <float> tempVector;
  numbers.clear();
  tempVector.clear();

  if(isCaloJet){

    Float_t SumCaloVsPtInitial[etaBins_];
    Float_t SumCaloVsPt[etaBins_];
    Float_t SumCaloPt[etaBins_];

    Float_t SumSquaredCaloVsPtInitial[etaBins_];
    Float_t SumSquaredCaloVsPt[etaBins_];
    Float_t SumSquaredCaloPt[etaBins_];

    // Need to set up histograms to get the RMS values for each pT bin 
    TH1F *hSumCaloVsPtInitial[nedge_pseudorapidity-1], *hSumCaloVsPt[nedge_pseudorapidity-1], *hSumCaloPt[nedge_pseudorapidity-1];

    for(int i = 0;i<etaBins_;++i)
      {

        SumCaloVsPtInitial[i] = 0;
        SumCaloVsPt[i] = 0;
        SumCaloPt[i] = 0;
        SumSquaredCaloVsPtInitial[i] = 0;
        SumSquaredCaloVsPt[i] = 0;
        SumSquaredCaloPt[i] = 0;

        hSumCaloVsPtInitial[i] = new TH1F(Form("hSumCaloVsPtInitial_%d",i),"",10000,-10000,10000);
        hSumCaloVsPt[i] = new TH1F(Form("hSumCaloVsPt_%d",i),"",10000,-10000,10000);
        hSumCaloPt[i] = new TH1F(Form("hSumCaloPt_%d",i),"",10000,-10000,10000);

      }

    for(unsigned icand = 0;icand<caloCandidates->size(); icand++){

      const CaloTower & tower  = (*caloCandidates)[icand];
      reco::CandidateViewRef ref(calocandidates_,icand);
      //10 is tower pT min
      if(tower.p4(vtx).Et() < 0.1) continue;

      vsPt = 0;
      vsPtInitial = 0;
      vsArea = 0;

      if(std::string("Vs")==UEAlgo) {
	const reco::VoronoiBackground& voronoi = (*VsBackgrounds)[ref];
	vsPt = voronoi.pt();
	vsPtInitial = voronoi.pt_subtracted();
	vsArea = voronoi.area();
      }

      NCaloTower++;

      caloPt = tower.p4(vtx).Et();
      caloEta = tower.p4(vtx).Eta();
      caloPhi = tower.p4(vtx).Phi();


      for(size_t k = 0;k<nedge_pseudorapidity-1; k++){
	if(caloEta >= edge_pseudorapidity[k] && caloEta < edge_pseudorapidity[k+1]){
	  SumCaloVsPtInitial[k] = SumCaloVsPtInitial[k] + vsPtInitial;
	  SumCaloVsPt[k] = SumCaloVsPt[k] + vsPt;
	  SumCaloPt[k] = SumCaloPt[k] + caloPt;
	  SumSquaredCaloVsPtInitial[k] = SumSquaredCaloVsPtInitial[k] + vsPtInitial*vsPtInitial;
	  SumSquaredCaloVsPt[k] = SumSquaredCaloVsPt[k] + vsPt*vsPt;
	  SumSquaredCaloPt[k] = SumSquaredCaloPt[k] + caloPt*caloPt;
	  break;
	}// eta selection statement 

      }// eta bin loop

      SumPt_value = SumPt_value + caloPt;

      mCaloPt->Fill(caloPt);
      mCaloEta->Fill(caloEta);
      mCaloPhi->Fill(caloPhi);
      mCaloVsPt->Fill(vsPt);
      mCaloVsPtInitial->Fill(vsPtInitial);
      mCaloArea->Fill(vsArea);

    }// calo tower candidate  loop

    for(int k = 0;k<nedge_pseudorapidity-1;k++){

      hSumCaloVsPtInitial[k]->Fill(SumCaloVsPtInitial[k]);
      hSumCaloVsPt[k]->Fill(SumCaloVsPt[k]);
      hSumCaloPt[k]->Fill(SumCaloPt[k]);

    }// eta bin loop  

    Float_t Evt_SumCaloVsPt = 0;
    Float_t Evt_SumCaloVsPtInitial = 0;
    Float_t Evt_SumCaloPt = 0;

    Float_t Evt_SumSquaredCaloVsPt = 0;
    Float_t Evt_SumSquaredCaloVsPtInitial = 0;
    Float_t Evt_SumSquaredCaloPt = 0;


    mSumCaloVsPtInitial_n5p191_n2p650->Fill(SumCaloVsPtInitial[0]);
    mSumCaloVsPtInitial_n2p650_n2p043->Fill(SumCaloVsPtInitial[1]);
    mSumCaloVsPtInitial_n2p043_n1p740->Fill(SumCaloVsPtInitial[2]);
    mSumCaloVsPtInitial_n1p740_n1p479->Fill(SumCaloVsPtInitial[3]);
    mSumCaloVsPtInitial_n1p479_n1p131->Fill(SumCaloVsPtInitial[4]);
    mSumCaloVsPtInitial_n1p131_n0p783->Fill(SumCaloVsPtInitial[5]);
    mSumCaloVsPtInitial_n0p783_n0p522->Fill(SumCaloVsPtInitial[6]);
    mSumCaloVsPtInitial_n0p522_0p522->Fill(SumCaloVsPtInitial[7]);
    mSumCaloVsPtInitial_0p522_0p783->Fill(SumCaloVsPtInitial[8]);
    mSumCaloVsPtInitial_0p783_1p131->Fill(SumCaloVsPtInitial[9]);
    mSumCaloVsPtInitial_1p131_1p479->Fill(SumCaloVsPtInitial[10]);
    mSumCaloVsPtInitial_1p479_1p740->Fill(SumCaloVsPtInitial[11]);
    mSumCaloVsPtInitial_1p740_2p043->Fill(SumCaloVsPtInitial[12]);
    mSumCaloVsPtInitial_2p043_2p650->Fill(SumCaloVsPtInitial[13]);
    mSumCaloVsPtInitial_2p650_5p191->Fill(SumCaloVsPtInitial[14]);

    mSumCaloVsPt_n5p191_n2p650->Fill(SumCaloVsPt[0]);
    mSumCaloVsPt_n2p650_n2p043->Fill(SumCaloVsPt[1]);
    mSumCaloVsPt_n2p043_n1p740->Fill(SumCaloVsPt[2]);
    mSumCaloVsPt_n1p740_n1p479->Fill(SumCaloVsPt[3]);
    mSumCaloVsPt_n1p479_n1p131->Fill(SumCaloVsPt[4]);
    mSumCaloVsPt_n1p131_n0p783->Fill(SumCaloVsPt[5]);
    mSumCaloVsPt_n0p783_n0p522->Fill(SumCaloVsPt[6]);
    mSumCaloVsPt_n0p522_0p522->Fill(SumCaloVsPt[7]);
    mSumCaloVsPt_0p522_0p783->Fill(SumCaloVsPt[8]);
    mSumCaloVsPt_0p783_1p131->Fill(SumCaloVsPt[9]);
    mSumCaloVsPt_1p131_1p479->Fill(SumCaloVsPt[10]);
    mSumCaloVsPt_1p479_1p740->Fill(SumCaloVsPt[11]);
    mSumCaloVsPt_1p740_2p043->Fill(SumCaloVsPt[12]);
    mSumCaloVsPt_2p043_2p650->Fill(SumCaloVsPt[13]);
    mSumCaloVsPt_2p650_5p191->Fill(SumCaloVsPt[14]);

    mSumCaloPt_n5p191_n2p650->Fill(SumCaloPt[0]);
    mSumCaloPt_n2p650_n2p043->Fill(SumCaloPt[1]);
    mSumCaloPt_n2p043_n1p740->Fill(SumCaloPt[2]);
    mSumCaloPt_n1p740_n1p479->Fill(SumCaloPt[3]);
    mSumCaloPt_n1p479_n1p131->Fill(SumCaloPt[4]);
    mSumCaloPt_n1p131_n0p783->Fill(SumCaloPt[5]);
    mSumCaloPt_n0p783_n0p522->Fill(SumCaloPt[6]);
    mSumCaloPt_n0p522_0p522->Fill(SumCaloPt[7]);
    mSumCaloPt_0p522_0p783->Fill(SumCaloPt[8]);
    mSumCaloPt_0p783_1p131->Fill(SumCaloPt[9]);
    mSumCaloPt_1p131_1p479->Fill(SumCaloPt[10]);
    mSumCaloPt_1p479_1p740->Fill(SumCaloPt[11]);
    mSumCaloPt_1p740_2p043->Fill(SumCaloPt[12]);
    mSumCaloPt_2p043_2p650->Fill(SumCaloPt[13]);
    mSumCaloPt_2p650_5p191->Fill(SumCaloPt[14]);


    for(size_t  k = 0;k<nedge_pseudorapidity-1;k++){

      Evt_SumCaloVsPtInitial = Evt_SumCaloVsPtInitial + SumCaloVsPtInitial[k];
      Evt_SumCaloVsPt = Evt_SumCaloVsPt + SumCaloVsPt[k];
      Evt_SumCaloPt = Evt_SumCaloPt + SumCaloPt[k];

      mSumCaloVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumCaloVsPtInitial[k]);
      mSumCaloVsPt_eta->Fill(edge_pseudorapidity[k],SumCaloVsPt[k]);
      mSumCaloPt_eta->Fill(edge_pseudorapidity[k],SumCaloPt[k]);

      Evt_SumSquaredCaloVsPtInitial = Evt_SumSquaredCaloVsPtInitial + SumSquaredCaloVsPtInitial[k];
      Evt_SumSquaredCaloVsPt = Evt_SumSquaredCaloVsPt + SumSquaredCaloVsPt[k];
      Evt_SumSquaredCaloPt = Evt_SumSquaredCaloPt + SumSquaredCaloPt[k];

      mSumSquaredCaloVsPtInitial_eta->Fill(edge_pseudorapidity[k],hSumCaloVsPtInitial[k]->GetRMS(1));
      mSumSquaredCaloVsPt_eta->Fill(edge_pseudorapidity[k],hSumCaloVsPt[k]->GetRMS(1));
      mSumSquaredCaloPt_eta->Fill(edge_pseudorapidity[k],hSumCaloPt[k]->GetRMS(1));

      delete hSumCaloVsPtInitial[k];
      delete hSumCaloVsPt[k];
      delete hSumCaloPt[k];

    }// eta bin loop  

    mSumCaloVsPtInitial->Fill(Evt_SumCaloVsPtInitial);
    mSumCaloVsPt->Fill(Evt_SumCaloVsPt);
    mSumCaloPt->Fill(Evt_SumCaloPt);
    mSumCaloVsPtInitial_HF->Fill(Evt_SumCaloVsPtInitial,HF_energy);
    mSumCaloVsPt_HF->Fill(Evt_SumCaloVsPt,HF_energy);
    mSumCaloPt_HF->Fill(Evt_SumCaloPt,HF_energy);

    mSumSquaredCaloVsPtInitial->Fill(Evt_SumSquaredCaloVsPtInitial);
    mSumSquaredCaloVsPt->Fill(Evt_SumSquaredCaloVsPt);
    mSumSquaredCaloPt->Fill(Evt_SumSquaredCaloPt);

    mNCalopart->Fill(NCaloTower);
    mSumpt->Fill(SumPt_value);

  }// is calo jet 

  if(isPFJet){

    Float_t SumPFVsPtInitial[etaBins_];
    Float_t SumPFVsPt[etaBins_];
    Float_t SumPFPt[etaBins_];

    Float_t SumSquaredPFVsPtInitial[etaBins_];
    Float_t SumSquaredPFVsPt[etaBins_];
    Float_t SumSquaredPFPt[etaBins_];

    // Need to set up histograms to get the RMS values for each pT bin 
    TH1F *hSumPFVsPtInitial[nedge_pseudorapidity-1], *hSumPFVsPt[nedge_pseudorapidity-1], *hSumPFPt[nedge_pseudorapidity-1];

    for(int i = 0;i<etaBins_;i++){

      SumPFVsPtInitial[i] = 0;
      SumPFVsPt[i] = 0;
      SumPFPt[i] = 0;
      SumSquaredPFVsPtInitial[i] = 0;
      SumSquaredPFVsPt[i] = 0;
      SumSquaredPFPt[i] = 0;

      hSumPFVsPtInitial[i] = new TH1F(Form("hSumPFVsPtInitial_%d",i),"",10000,-10000,10000);
      hSumPFVsPt[i] = new TH1F(Form("hSumPFVsPt_%d",i),"",10000,-10000,10000);
      hSumPFPt[i] = new TH1F(Form("hSumPFPt_%d",i),"",10000,-10000,10000);

    }

    vector<vector <float> > PF_Space(1,vector<float>(3)); 

    for(unsigned icand=0;icand<pfCandidateColl->size(); icand++)
      {
        const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
        reco::CandidateViewRef ref(pfcandidates_,icand);
        if(pfCandidate.pt() < 5) continue;

        if(std::string("Vs")==UEAlgo) 
	  {

	    const reco::VoronoiBackground& voronoi = (*VsBackgrounds)[ref];
	    vsPt = voronoi.pt();
	    vsPtInitial = voronoi.pt_subtracted();
	    vsArea = voronoi.area();

	  }

        NPFpart++;
        pfPt = pfCandidate.pt();
        pfEta = pfCandidate.eta();
        pfPhi = pfCandidate.phi();
	pfID = pfCandidate.particleId();

	bool isBarrel = false; 
	bool isEndcap = false; 
	bool isForward = false; 

	if(fabs(pfEta)<BarrelEta) isBarrel = true;
	if(fabs(pfEta)>=BarrelEta && fabs(pfEta)<EndcapEta) isEndcap = true;
	if(fabs(pfEta)>=EndcapEta && fabs(pfEta)<ForwardEta) isForward = true;
	
	switch(pfID){
	case 0 :
	  mPFCandpT_vs_eta_Unknown->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_Unknown->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_Unknown->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_Unknown->Fill(pfPt);
	case 1 :
	  mPFCandpT_vs_eta_ChargedHadron->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_ChargedHadron->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_ChargedHadron->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_ChargedHadron->Fill(pfPt);
	case 2 :
	  mPFCandpT_vs_eta_electron->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_electron->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_electron->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_electron->Fill(pfPt);
	case 3 :
	  mPFCandpT_vs_eta_muon->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_muon->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_muon->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_muon->Fill(pfPt);
	case 4 :
	  mPFCandpT_vs_eta_photon->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_photon->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_photon->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_photon->Fill(pfPt);
	case 5 :
	  mPFCandpT_vs_eta_NeutralHadron->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_NeutralHadron->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_NeutralHadron->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_NeutralHadron->Fill(pfPt);
	case 6 :
	  mPFCandpT_vs_eta_HadE_inHF->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_HadE_inHF->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_HadE_inHF->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_HadE_inHF->Fill(pfPt);
	case 7 :
	  mPFCandpT_vs_eta_EME_inHF->Fill(pfPt, pfEta);
	  if(isBarrel) mPFCandpT_Barrel_EME_inHF->Fill(pfPt);
	  if(isEndcap) mPFCandpT_Endcap_EME_inHF->Fill(pfPt);
	  if(isForward) mPFCandpT_Forward_EME_inHF->Fill(pfPt);
	}
	
	
        //Fill 2d vector matrix
        tempVector.push_back(pfPt);
        tempVector.push_back(pfEta);
        tempVector.push_back(pfPhi);
        tempVector.push_back(vsPtInitial);
        tempVector.push_back(vsPt);


        numbers.push_back(tempVector);
        tempVector.clear();

        for(size_t k = 0;k<nedge_pseudorapidity-1; k++)
	  {
	    if(pfEta >= edge_pseudorapidity[k] && pfEta < edge_pseudorapidity[k+1]){
	      SumPFVsPtInitial[k] = SumPFVsPtInitial[k] + vsPtInitial;
	      SumPFVsPt[k] = SumPFVsPt[k] + vsPt;
	      SumPFPt[k] = SumPFPt[k] + pfPt;

	      SumSquaredPFVsPtInitial[k] = SumSquaredPFVsPtInitial[k] + vsPtInitial*vsPtInitial;
	      SumSquaredPFVsPt[k] = SumSquaredPFVsPt[k] + vsPt*vsPt;
	      SumSquaredPFPt[k] = SumSquaredPFPt[k] + pfPt*pfPt;
	      break;
	    }// eta selection statement 

	  }// eta bin loop

        SumPt_value = SumPt_value + pfPt;

        mPFPt->Fill(pfPt);
        mPFEta->Fill(pfEta);
        mPFPhi->Fill(pfPhi);
        mPFVsPt->Fill(vsPt);
        mPFVsPtInitial->Fill(vsPtInitial); //filled with zeros (need a fix)
        mPFVsPtInitial_HF->Fill(cent->EtHFtowerSum(), vsPtInitial,1); //filled with zeros (need a fix)

        if(isBarrel)
	  {
	    if(hibin >= 190 && hibin < 200)
	      mPFVsPtInitial_Barrel_Centrality_100_95->Fill(vsPtInitial);

	    if(hibin >= 180 && hibin < 190)
	      mPFVsPtInitial_Barrel_Centrality_95_90->Fill(vsPtInitial);

	    if(hibin >= 170 && hibin < 180)
	      mPFVsPtInitial_Barrel_Centrality_90_85->Fill(vsPtInitial);

	    if(hibin >= 160 && hibin < 170)
	      mPFVsPtInitial_Barrel_Centrality_85_80->Fill(vsPtInitial);

	    if(hibin >= 150 && hibin < 160)
	      mPFVsPtInitial_Barrel_Centrality_80_75->Fill(vsPtInitial);

	    if(hibin >= 140 && hibin < 150)
	      mPFVsPtInitial_Barrel_Centrality_75_70->Fill(vsPtInitial);

	    if(hibin >= 130 && hibin < 140)
	      mPFVsPtInitial_Barrel_Centrality_70_65->Fill(vsPtInitial);

	    if(hibin >= 120 && hibin < 130)
	      mPFVsPtInitial_Barrel_Centrality_65_60->Fill(vsPtInitial);

	    if(hibin >= 110 && hibin < 120) 
	      mPFVsPtInitial_Barrel_Centrality_60_55->Fill(vsPtInitial);

	    if(hibin >= 100 && hibin < 110)
	      mPFVsPtInitial_Barrel_Centrality_55_50->Fill(vsPtInitial);

	    if(hibin >= 90 && hibin < 100)
	      mPFVsPtInitial_Barrel_Centrality_50_45->Fill(vsPtInitial);

	    if(hibin >= 80 && hibin < 90)
	      mPFVsPtInitial_Barrel_Centrality_45_40->Fill(vsPtInitial);

	    if(hibin >= 70 && hibin < 80)
	      mPFVsPtInitial_Barrel_Centrality_40_35->Fill(vsPtInitial);

	    if(hibin >= 60 && hibin < 70)
	      mPFVsPtInitial_Barrel_Centrality_35_30->Fill(vsPtInitial);

	    if(hibin >= 50 && hibin < 60)
	      mPFVsPtInitial_Barrel_Centrality_30_25->Fill(vsPtInitial);

	    if(hibin >= 40 && hibin < 50)
	      mPFVsPtInitial_Barrel_Centrality_25_20->Fill(vsPtInitial);

	    if(hibin >= 30 && hibin < 40)
	      mPFVsPtInitial_Barrel_Centrality_20_15->Fill(vsPtInitial);

	    if(hibin >= 20 && hibin < 30)
	      mPFVsPtInitial_Barrel_Centrality_15_10->Fill(vsPtInitial);

	    if(hibin >= 10 && hibin < 20)
	      mPFVsPtInitial_Barrel_Centrality_10_5->Fill(vsPtInitial);

	    if(hibin >= 0 && hibin < 10)
	      mPFVsPtInitial_Barrel_Centrality_5_0->Fill(vsPtInitial);
	  }


        if(isEndcap)
	  {

	    if(hibin >= 190 && hibin < 200)
	      mPFVsPtInitial_EndCap_Centrality_100_95->Fill(vsPtInitial);

	    if(hibin >= 180 && hibin < 190)
	      mPFVsPtInitial_EndCap_Centrality_95_90->Fill(vsPtInitial);

	    if(hibin >= 170 && hibin < 180)
	      mPFVsPtInitial_EndCap_Centrality_90_85->Fill(vsPtInitial);

	    if(hibin >= 160 && hibin < 170)
	      mPFVsPtInitial_EndCap_Centrality_85_80->Fill(vsPtInitial);

	    if(hibin >= 150 && hibin < 160)
	      mPFVsPtInitial_EndCap_Centrality_80_75->Fill(vsPtInitial);

	    if(hibin >= 140 && hibin < 150)
	      mPFVsPtInitial_EndCap_Centrality_75_70->Fill(vsPtInitial);

	    if(hibin >= 130 && hibin < 140)
	      mPFVsPtInitial_EndCap_Centrality_70_65->Fill(vsPtInitial);

	    if(hibin >= 120 && hibin < 130)
	      mPFVsPtInitial_EndCap_Centrality_65_60->Fill(vsPtInitial);

	    if(hibin >= 110 && hibin < 120) 
	      mPFVsPtInitial_EndCap_Centrality_60_55->Fill(vsPtInitial);

	    if(hibin >= 100 && hibin < 110)
	      mPFVsPtInitial_EndCap_Centrality_55_50->Fill(vsPtInitial);

	    if(hibin >= 90 && hibin < 100)
	      mPFVsPtInitial_EndCap_Centrality_50_45->Fill(vsPtInitial);

	    if(hibin >= 80 && hibin < 90)
	      mPFVsPtInitial_EndCap_Centrality_45_40->Fill(vsPtInitial);

	    if(hibin >= 70 && hibin < 80)
	      mPFVsPtInitial_EndCap_Centrality_40_35->Fill(vsPtInitial);

	    if(hibin >= 60 && hibin < 70)
	      mPFVsPtInitial_EndCap_Centrality_35_30->Fill(vsPtInitial);

	    if(hibin >= 50 && hibin < 60)
	      mPFVsPtInitial_EndCap_Centrality_30_25->Fill(vsPtInitial);

	    if(hibin >= 40 && hibin < 50)
	      mPFVsPtInitial_EndCap_Centrality_25_20->Fill(vsPtInitial);

	    if(hibin >= 30 && hibin < 40)
	      mPFVsPtInitial_EndCap_Centrality_20_15->Fill(vsPtInitial);

	    if(hibin >= 20 && hibin < 30)
	      mPFVsPtInitial_EndCap_Centrality_15_10->Fill(vsPtInitial);

	    if(hibin >= 10 && hibin < 20)
	      mPFVsPtInitial_EndCap_Centrality_10_5->Fill(vsPtInitial);

	    if(hibin >= 0 && hibin < 10)
	      mPFVsPtInitial_EndCap_Centrality_5_0->Fill(vsPtInitial);
	    
	  }
    
     
        if(isForward)
	  {


	    if(hibin >= 190 && hibin < 200)
	      mPFVsPtInitial_HF_Centrality_100_95->Fill(vsPtInitial);

	    if(hibin >= 180 && hibin < 190)
	      mPFVsPtInitial_HF_Centrality_95_90->Fill(vsPtInitial);

	    if(hibin >= 170 && hibin < 180)
	      mPFVsPtInitial_HF_Centrality_90_85->Fill(vsPtInitial);

	    if(hibin >= 160 && hibin < 170)
	      mPFVsPtInitial_HF_Centrality_85_80->Fill(vsPtInitial);

	    if(hibin >= 150 && hibin < 160)
	      mPFVsPtInitial_HF_Centrality_80_75->Fill(vsPtInitial);

	    if(hibin >= 140 && hibin < 150)
	      mPFVsPtInitial_HF_Centrality_75_70->Fill(vsPtInitial);

	    if(hibin >= 130 && hibin < 140)
	      mPFVsPtInitial_HF_Centrality_70_65->Fill(vsPtInitial);

	    if(hibin >= 120 && hibin < 130)
	      mPFVsPtInitial_HF_Centrality_65_60->Fill(vsPtInitial);

	    if(hibin >= 110 && hibin < 120) 
	      mPFVsPtInitial_HF_Centrality_60_55->Fill(vsPtInitial);

	    if(hibin >= 100 && hibin < 110)
	      mPFVsPtInitial_HF_Centrality_55_50->Fill(vsPtInitial);

	    if(hibin >= 90 && hibin < 100)
	      mPFVsPtInitial_HF_Centrality_50_45->Fill(vsPtInitial);

	    if(hibin >= 80 && hibin < 90)
	      mPFVsPtInitial_HF_Centrality_45_40->Fill(vsPtInitial);

	    if(hibin >= 70 && hibin < 80)
	      mPFVsPtInitial_HF_Centrality_40_35->Fill(vsPtInitial);

	    if(hibin >= 60 && hibin < 70)
	      mPFVsPtInitial_HF_Centrality_35_30->Fill(vsPtInitial);

	    if(hibin >= 50 && hibin < 60)
	      mPFVsPtInitial_HF_Centrality_30_25->Fill(vsPtInitial);

	    if(hibin >= 40 && hibin < 50)
	      mPFVsPtInitial_HF_Centrality_25_20->Fill(vsPtInitial);

	    if(hibin >= 30 && hibin < 40)
	      mPFVsPtInitial_HF_Centrality_20_15->Fill(vsPtInitial);

	    if(hibin >= 20 && hibin < 30)
	      mPFVsPtInitial_HF_Centrality_15_10->Fill(vsPtInitial);

	    if(hibin >= 10 && hibin < 20)
	      mPFVsPtInitial_HF_Centrality_10_5->Fill(vsPtInitial);

	    if(hibin >= 0 && hibin < 10)
	      mPFVsPtInitial_HF_Centrality_5_0->Fill(vsPtInitial);

	    
	  }




	mPFArea->Fill(vsArea);



      }// pf candidate loop 

    for(int k = 0;k<nedge_pseudorapidity-1;k++){

      hSumPFVsPtInitial[k]->Fill(SumPFVsPtInitial[k]);
      hSumPFVsPt[k]->Fill(SumPFVsPt[k]);
      hSumPFPt[k]->Fill(SumPFPt[k]);

    }// eta bin loop  

    Float_t Evt_SumPFVsPt = 0;
    Float_t Evt_SumPFVsPtInitial = 0;
    Float_t Evt_SumPFPt = 0;
    Float_t Evt_SumSquaredPFVsPt = 0;
    Float_t Evt_SumSquaredPFVsPtInitial = 0;
    Float_t Evt_SumSquaredPFPt = 0;

    mSumPFVsPtInitial_n5p191_n2p650->Fill(SumPFVsPtInitial[0]);
    mSumPFVsPtInitial_n2p650_n2p043->Fill(SumPFVsPtInitial[1]);
    mSumPFVsPtInitial_n2p043_n1p740->Fill(SumPFVsPtInitial[2]);
    mSumPFVsPtInitial_n1p740_n1p479->Fill(SumPFVsPtInitial[3]);
    mSumPFVsPtInitial_n1p479_n1p131->Fill(SumPFVsPtInitial[4]);
    mSumPFVsPtInitial_n1p131_n0p783->Fill(SumPFVsPtInitial[5]);
    mSumPFVsPtInitial_n0p783_n0p522->Fill(SumPFVsPtInitial[6]);
    mSumPFVsPtInitial_n0p522_0p522->Fill(SumPFVsPtInitial[7]);
    mSumPFVsPtInitial_0p522_0p783->Fill(SumPFVsPtInitial[8]);
    mSumPFVsPtInitial_0p783_1p131->Fill(SumPFVsPtInitial[9]);
    mSumPFVsPtInitial_1p131_1p479->Fill(SumPFVsPtInitial[10]);
    mSumPFVsPtInitial_1p479_1p740->Fill(SumPFVsPtInitial[11]);
    mSumPFVsPtInitial_1p740_2p043->Fill(SumPFVsPtInitial[12]);
    mSumPFVsPtInitial_2p043_2p650->Fill(SumPFVsPtInitial[13]);
    mSumPFVsPtInitial_2p650_5p191->Fill(SumPFVsPtInitial[14]);

    mSumPFVsPt_n5p191_n2p650->Fill(SumPFVsPt[0]);
    mSumPFVsPt_n2p650_n2p043->Fill(SumPFVsPt[1]);
    mSumPFVsPt_n2p043_n1p740->Fill(SumPFVsPt[2]);
    mSumPFVsPt_n1p740_n1p479->Fill(SumPFVsPt[3]);
    mSumPFVsPt_n1p479_n1p131->Fill(SumPFVsPt[4]);
    mSumPFVsPt_n1p131_n0p783->Fill(SumPFVsPt[5]);
    mSumPFVsPt_n0p783_n0p522->Fill(SumPFVsPt[6]);
    mSumPFVsPt_n0p522_0p522->Fill(SumPFVsPt[7]);
    mSumPFVsPt_0p522_0p783->Fill(SumPFVsPt[8]);
    mSumPFVsPt_0p783_1p131->Fill(SumPFVsPt[9]);
    mSumPFVsPt_1p131_1p479->Fill(SumPFVsPt[10]);
    mSumPFVsPt_1p479_1p740->Fill(SumPFVsPt[11]);
    mSumPFVsPt_1p740_2p043->Fill(SumPFVsPt[12]);
    mSumPFVsPt_2p043_2p650->Fill(SumPFVsPt[13]);
    mSumPFVsPt_2p650_5p191->Fill(SumPFVsPt[14]);

    mSumPFPt_n5p191_n2p650->Fill(SumPFPt[0]);
    mSumPFPt_n2p650_n2p043->Fill(SumPFPt[1]);
    mSumPFPt_n2p043_n1p740->Fill(SumPFPt[2]);
    mSumPFPt_n1p740_n1p479->Fill(SumPFPt[3]);
    mSumPFPt_n1p479_n1p131->Fill(SumPFPt[4]);
    mSumPFPt_n1p131_n0p783->Fill(SumPFPt[5]);
    mSumPFPt_n0p783_n0p522->Fill(SumPFPt[6]);
    mSumPFPt_n0p522_0p522->Fill(SumPFPt[7]);
    mSumPFPt_0p522_0p783->Fill(SumPFPt[8]);
    mSumPFPt_0p783_1p131->Fill(SumPFPt[9]);
    mSumPFPt_1p131_1p479->Fill(SumPFPt[10]);
    mSumPFPt_1p479_1p740->Fill(SumPFPt[11]);
    mSumPFPt_1p740_2p043->Fill(SumPFPt[12]);
    mSumPFPt_2p043_2p650->Fill(SumPFPt[13]);
    mSumPFPt_2p650_5p191->Fill(SumPFPt[14]);

    for(size_t  k = 0;k<nedge_pseudorapidity-1;k++){

      Evt_SumPFVsPtInitial = Evt_SumPFVsPtInitial + SumPFVsPtInitial[k];
      Evt_SumPFVsPt = Evt_SumPFVsPt + SumPFVsPt[k];
      Evt_SumPFPt = Evt_SumPFPt + SumPFPt[k];

      mSumPFVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumPFVsPtInitial[k]);
      mSumPFVsPt_eta->Fill(edge_pseudorapidity[k],SumPFVsPt[k]);
      mSumPFPt_eta->Fill(edge_pseudorapidity[k],SumPFPt[k]);

      Evt_SumSquaredPFVsPtInitial = Evt_SumSquaredPFVsPtInitial + SumSquaredPFVsPtInitial[k];
      Evt_SumSquaredPFVsPt = Evt_SumSquaredPFVsPt + SumSquaredPFVsPt[k];
      Evt_SumSquaredPFPt = Evt_SumSquaredPFPt + SumSquaredPFPt[k];

      mSumSquaredPFVsPtInitial_eta->Fill(edge_pseudorapidity[k],hSumPFVsPtInitial[k]->GetRMS(1));
      mSumSquaredPFVsPt_eta->Fill(edge_pseudorapidity[k],hSumPFVsPt[k]->GetRMS(1));
      mSumSquaredPFPt_eta->Fill(edge_pseudorapidity[k],hSumPFPt[k]->GetRMS(1));

      delete hSumPFVsPtInitial[k];
      delete hSumPFVsPt[k];
      delete hSumPFPt[k];

    }// eta bin loop  

    mSumPFVsPtInitial->Fill(Evt_SumPFVsPtInitial);
    mSumPFVsPt->Fill(Evt_SumPFVsPt);
    mSumPFPt->Fill(Evt_SumPFPt);
    mSumPFVsPtInitial_HF->Fill(Evt_SumPFVsPtInitial,HF_energy);
    mSumPFVsPt_HF->Fill(Evt_SumPFVsPt,HF_energy);
    mSumPFPt_HF->Fill(Evt_SumPFPt,HF_energy);

    mSumSquaredPFVsPtInitial->Fill(Evt_SumSquaredPFVsPtInitial);
    mSumSquaredPFVsPt->Fill(Evt_SumSquaredPFVsPt);
    mSumSquaredPFPt->Fill(Evt_SumSquaredPFPt);

    mNPFpart->Fill(NPFpart);
    mSumpt->Fill(SumPt_value);

  }

  if (isCaloJet)
    {
      for (unsigned ijet=0; ijet<caloJets->size(); ijet++) 
	{
	  recoJets.push_back((*caloJets)[ijet]);
	} 
    }

  if (isJPTJet)
    {
      for (unsigned ijet=0; ijet<jptJets->size(); ijet++) 
        recoJets.push_back((*jptJets)[ijet]);
    }

  if (isPFJet) 
    {
      if(std::string("Pu")==UEAlgo)
	{
	  for (unsigned ijet=0; ijet<basicJets->size();ijet++) 
	    {
	      recoJets.push_back((*basicJets)[ijet]);
	    }
	}
      if(std::string("Vs")==UEAlgo){
        for (unsigned ijet=0; ijet<pfJets->size(); ijet++) 
	  {
	    recoJets.push_back((*pfJets)[ijet]);
	  }
      }
    }


  if (isCaloJet && !caloJets.isValid()) 
    {
      return;
    }
  if (isJPTJet  && !jptJets.isValid()) 
    {
      return;
    }
  if (isPFJet){
    if(std::string("Pu")==UEAlgo){if(!basicJets.isValid())   return;}
    if(std::string("Vs")==UEAlgo){if(!pfJets.isValid())   return;}
  }


  int nJet_40 = 0;

  mNJets->Fill(recoJets.size());

  for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
    if (recoJets[ijet].pt() > mRecoJetPtThreshold) {
      //counting forward and barrel jets
      // get an idea of no of jets with pT>40 GeV 
      if(recoJets[ijet].pt() > 40)
	nJet_40++;
      if (mEta) mEta->Fill(recoJets[ijet].eta());
      if (mjetpileup) mjetpileup->Fill(recoJets[ijet].pileup());
      if (mJetArea)      mJetArea     ->Fill(recoJets[ijet].jetArea());
      if (mPhi)          mPhi         ->Fill(recoJets[ijet].phi());
      if (mEnergy)       mEnergy      ->Fill(recoJets[ijet].energy());
      if (mP)            mP           ->Fill(recoJets[ijet].p());
      if (mPt)           mPt          ->Fill(recoJets[ijet].pt());
      if (mMass)         mMass        ->Fill(recoJets[ijet].mass());
      if (mConstituents) mConstituents->Fill(recoJets[ijet].nConstituents());

      for(size_t iii = 0 ; iii < numbers.size() ; iii++)
        {
          pfDeltaR = sqrt((numbers[iii][2]-recoJets[ijet].phi())*(numbers[iii][2]-recoJets[ijet].phi()) + (numbers[iii][1]-recoJets[ijet].eta())*(numbers[iii][1]-recoJets[ijet].eta())); //MZ

          mPFVsPtInitialDeltaR_pTCorrected->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ


          mPFVsPtDeltaR_pTCorrected->Fill(pfDeltaR,numbers[iii][4]/recoJets[ijet].pt()); //MZ

          //if(pfDeltaR < 1.0)         
          // mPFDeltaR ->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
          mPFDeltaR ->Fill(pfDeltaR); //MZ
          mPFDeltaR_Scaled_R->Fill(pfDeltaR,1. / pow(pfDeltaR,2)); //MZ
          //mPFDeltaR_pTCorrected->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
          mPFDeltaR_pTCorrected->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ

          //mPFVsPtInitialDeltaR_pTCorrected->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ

          if(recoJets[ijet].pt() > 20 && recoJets[ijet].pt() < 30)
	    {
	      mPFDeltaR_pTCorrected_PFpT_20To30->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_20To30->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }

          if(recoJets[ijet].pt() > 30 && recoJets[ijet].pt() < 50)
	    {
	      mPFDeltaR_pTCorrected_PFpT_30To50->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_30To50->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }

          if(recoJets[ijet].pt() > 50 && recoJets[ijet].pt() < 80)
	    {
	      mPFDeltaR_pTCorrected_PFpT_50To80->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_50To80->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }


          if(recoJets[ijet].pt() > 80 && recoJets[ijet].pt() < 120)
	    {
	      mPFDeltaR_pTCorrected_PFpT_80To120->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_80To120->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }


          if(recoJets[ijet].pt() > 120 && recoJets[ijet].pt() < 180)
	    {
	      mPFDeltaR_pTCorrected_PFpT_120To180->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_120To180->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }


          if(recoJets[ijet].pt() > 180 && recoJets[ijet].pt() < 300)
	    {
	      mPFDeltaR_pTCorrected_PFpT_180To300->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_180To300->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }

          if(recoJets[ijet].pt() > 300)
	    {
	      mPFDeltaR_pTCorrected_PFpT_300ToInf->Fill(pfDeltaR,numbers[iii][0]/recoJets[ijet].pt()); //MZ
	      mPFDeltaR_pTCorrected_PFVsInitialpT_300ToInf->Fill(pfDeltaR,numbers[iii][3]/recoJets[ijet].pt()); //MZ
	    }

        }

    }
    //cout << "End RecoJets loop" << endl; 
    if (mNJets_40) 
      mNJets_40->Fill(nJet_40); 

  } // end recojet loop 222

  numbers.clear();

}

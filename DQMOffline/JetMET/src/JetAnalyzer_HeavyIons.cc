//
// Jet Analyzer class for heavy ion jets. for DQM jet analysis monitoring 
// For CMSSW_7_4_X, especially reading background subtracted jets 
// author: Raghav Kunnawalkam Elayavalli,
//         Jan 12th 2015 
//         Rutgers University, email: raghav.k.e at CERN dot CH 
//


#include "DQMOffline/JetMET/interface/JetAnalyzer_HeavyIons.h"

using namespace edm;
using namespace reco;
using namespace std;

// declare the constructors:

JetAnalyzer_HeavyIons::JetAnalyzer_HeavyIons(const edm::ParameterSet& iConfig) :
  mInputCollection               (iConfig.getParameter<edm::InputTag>       ("src")),
  mInputPFCandCollection         (iConfig.getParameter<edm::InputTag>       ("PFcands")),
//  rhoTag                         (iConfig.getParameter<edm::InputTag>       ("srcRho")),
  centrality                     (iConfig.getParameter<edm::InputTag>       ("centrality")),
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

  // std::size_t foundCaloCollection = inputCollectionLabel.find("Calo");
  // std::size_t foundJPTCollection  = inputCollectionLabel.find("JetPlusTrack");
  // std::size_t foundPFCollection   = inputCollectionLabel.find("PF");

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
  centralityToken_ = consumes<reco::Centrality>(centrality);
  hiVertexToken_ = consumes<std::vector<reco::Vertex> >(edm::InputTag("hiSelectedVertex"));
  
  // need to initialize the PF cand histograms : which are also event variables 
  if(isPFJet){
    mNPFpart = 0;
    mPFPt = 0;
    mPFEta = 0;
    mPFPhi = 0;
    mPFVsPt = 0;
    mPFVsPtInitial = 0;
    mPFArea = 0;
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
    mPFVsPtInitial_eta_phi = 0;
    mPFVsPt_eta_phi = 0;
    mPFPt_eta_phi = 0;

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
    mCaloVsPtInitial_eta_phi = 0;
    mCaloVsPt_eta_phi = 0;
    mCaloPt_eta_phi = 0;

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
  mvn = 0;
  mpsin = 0;
  


  // Events variables
  mNvtx         = 0;
  mHF           = 0;

  // added Jan 12th 2015

  mDeltapT = 0;
  //mDeltapT_HF = 0;
  mDeltapT_eta = 0;
  //mDeltapT_phiMinusPsi2 = 0;
  mDeltapT_eta_phi = 0;
  
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

  mVs_0_x = 0;
  mVs_0_y = 0;
  mVs_1_x = 0;
  mVs_1_y = 0;
  mVs_2_x = 0;
  mVs_2_y = 0;
  mVs_0_x_versus_HF = 0;
  mVs_0_y_versus_HF = 0;
  mVs_1_x_versus_HF = 0;
  mVs_1_y_versus_HF = 0;
  mVs_2_x_versus_HF = 0;
  mVs_2_y_versus_HF = 0;

}
   
void JetAnalyzer_HeavyIons::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &) 
  {

    ibooker.setCurrentFolder("JetMET/JetValidation/"+mInputCollection.label());
    
    static const size_t ncms_hcal_edge_pseudorapidity = 82 + 1;
    static const double cms_hcal_edge_pseudorapidity[ncms_hcal_edge_pseudorapidity] = {
      -5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013,
      -3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853,
      -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830,
      -1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218,
      -1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609,
      -0.522, -0.435, -0.348, -0.261, -0.174, -0.087,
      0.000,
      0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609,
      0.696, 0.783, 0.879, 0.957, 1.044, 1.131, 1.218,
      1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830,
      1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853,
      2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013,
      4.191, 4.363, 4.538, 4.716, 4.889, 5.191
    };
    
    //cout<<"inside the book histograms function"<<endl;

    double edge_pseudorapidity[etaBins_ +1] = {-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191 };

    TH2F *h2D_etabins_vs_pt2 = new TH2F("h2D_etabins_vs_pt2","etaBins (x axis), sum pt^{2} (y axis)",etaBins_,edge_pseudorapidity,10000,0,10000);
    TH2F *h2D_etabins_vs_pt = new TH2F("h2D_etabins_vs_pt","etaBins (x axis), sum pt (y axis)",etaBins_,edge_pseudorapidity,10000,-1000,1000);
    TH2F *h2D_etabins_vs_phi = new TH2F("h2D_etabins_vs_phi","candidate map, eta(x axis), phi (y axis), pt (z axis)",ncms_hcal_edge_pseudorapidity-1,cms_hcal_edge_pseudorapidity,36,-TMath::Pi(),TMath::Pi());

    if(isPFJet){

      mNPFpart         = ibooker.book1D("NPFpart","No of particle flow candidates",1000,0,10000);
      mPFPt            = ibooker.book1D("PFPt","PF candidate p_{T}",1000,-5000,5000);
      mPFEta           = ibooker.book1D("PFEta","PF candidate #eta",120,-6,6);
      mPFPhi           = ibooker.book1D("PFPhi","PF candidate #phi",70,-3.5,3.5);
      mPFVsPt          = ibooker.book1D("PFVsPt","Vs PF candidate p_{T}",1000,-5000,5000);
      mPFVsPtInitial   = ibooker.book1D("PFVsPtInitial","Vs background subtracted PF candidate p_{T}",1000,-5000,5000);
      mPFArea          = ibooker.book1D("PFArea","VS PF candidate area",100,0,4);
      
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
      mPFVsPtInitial_eta_phi    = ibooker.book2D("PFVsPtInitial_eta_phi",h2D_etabins_vs_phi);
      mPFVsPt_eta_phi    = ibooker.book2D("PFVsPt_eta_phi",h2D_etabins_vs_phi);
      mPFPt_eta_phi    = ibooker.book2D("PFPt_eta_phi",h2D_etabins_vs_phi);

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
     
      
    }

    if(isCaloJet){

      mNCalopart         = ibooker.book1D("NCalopart","No of particle flow candidates",1000,0,10000);
      mCaloPt            = ibooker.book1D("CaloPt","Calo candidate p_{T}",1000,-5000,5000);
      mCaloEta           = ibooker.book1D("CaloEta","Calo candidate #eta",120,-6,6);
      mCaloPhi           = ibooker.book1D("CaloPhi","Calo candidate #phi",70,-3.5,3.5);
      mCaloVsPt          = ibooker.book1D("CaloVsPt","Vs Calo candidate p_{T}",1000,-5000,5000);
      mCaloVsPtInitial   = ibooker.book1D("CaloVsPtInitial","Vs background subtracted Calo candidate p_{T}",1000,-5000,5000);
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
      mCaloVsPtInitial_eta_phi    = ibooker.book2D("CaloVsPtInitial_eta_phi",h2D_etabins_vs_phi);
      mCaloVsPt_eta_phi    = ibooker.book2D("CaloVsPt_eta_phi",h2D_etabins_vs_phi);
      mCaloPt_eta_phi    = ibooker.book2D("CaloPt_eta_phi",h2D_etabins_vs_phi);

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
    mvn              = ibooker.book1D("vn","vn",100,0,10);
    mpsin            = ibooker.book1D("mpsin","psin",100,0,10);

    // Event variables
    mNvtx            = ibooker.book1D("Nvtx",           "number of vertices", 60, 0, 60);
    mHF              = ibooker.book1D("HF", "HF energy distribution",1000,0,10000);

    // added Jan 12th 2015
    mDeltapT  = ibooker.book1D("DeltapT","",400,-200,200);
    //mDeltapT_HF  = ibooker.book2D("DeltapT_HF","",400,-200,200,1000,0,10000);
    mDeltapT_eta = ibooker.book2D("DeltapT_eta","",60,-6,+6,400,-200,200);
    //mDeltapT_phiMinusPsi2 = ibooker.book2D("DeltapT_phiMinusPsi2","",400,-200,200,35,-1.75,1.75);
    mDeltapT_eta_phi = ibooker.book2D("DeltapT_eta_phi",h2D_etabins_vs_phi);
    
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
    mNJets_40        = ibooker.book1D("NJets_pt_greater_40", "NJets pT > 40 GeV",  50,    0,   100);
    mNJets        = ibooker.book1D("NJets", "NJets",  50,    0,   100);

    mVs_0_x  = ibooker.book1D("Vs_0_x","flow modulated sumpT from both HF (+ and -) with cos, vn = 0",1000,-10000,10000);
    mVs_0_y  = ibooker.book1D("Vs_0_y","flow modulated sumpT from both HF (+ and -) with sin, vn = 0",1000,-10000,10000);
    mVs_1_x  = ibooker.book1D("Vs_1_x","flow modulated sumpT from both HF (+ and -) with cos, vn = 1",1000,-10000,10000);
    mVs_1_y  = ibooker.book1D("Vs_1_y","flow modulated sumpT from both HF (+ and -) with sin, vn = 1",1000,-10000,10000);
    mVs_2_x  = ibooker.book1D("Vs_2_x","flow modulated sumpT from both HF (+ and -) with cos, vn = 2",1000,-10000,10000);
    mVs_2_y  = ibooker.book1D("Vs_2_y","flow modulated sumpT from both HF (+ and -) with sin, vn = 2",1000,-10000,10000);
    
    mVs_0_x_versus_HF = ibooker.book2D("Vs_0_x_versus_HF","flow modulated sumpT (y axis) from both HF (+ and -) with cos, vn = 0 versus HF energy (x axis)",1000,0,10000,1000,-10000,10000);
    mVs_0_y_versus_HF = ibooker.book2D("Vs_0_y_versus_HF","flow modulated sumpT (y axis) from both HF (+ and -) with sin, vn = 0 versus HF energy (x axis)",1000,0,10000,1000,-10000,10000);
    mVs_1_x_versus_HF = ibooker.book2D("Vs_1_x_versus_HF","flow modulated sumpT (y axis) from both HF (+ and -) with cos, vn = 1 versus HF energy (x axis)",1000,0,10000,1000,-10000,10000);
    mVs_1_y_versus_HF = ibooker.book2D("Vs_1_y_versus_HF","flow modulated sumpT (y axis) from both HF (+ and -) with sin, vn = 1 versus HF energy (x axis)",1000,0,10000,1000,-10000,10000);
    mVs_2_x_versus_HF = ibooker.book2D("Vs_2_x_versus_HF","flow modulated sumpT (y axis) from both HF (+ and -) with cos, vn = 2 versus HF energy (x axis)",1000,0,10000,1000,-10000,10000);
    mVs_2_y_versus_HF = ibooker.book2D("Vs_2_y_versus_HF","flow modulated sumpT (y axis) from both HF (+ and -) with sin, vn = 2 versus HF energy (x axis)",1000,0,10000,1000,-10000,10000);
    
    if (mOutputFile.empty ()) 
      LogInfo("OutputInfo") << " Histograms will NOT be saved";
    else 
      LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;
  }



//------------------------------------------------------------------------------
// ~JetAnalyzer_HeavyIons
//------------------------------------------------------------------------------
JetAnalyzer_HeavyIons::~JetAnalyzer_HeavyIons() {}


//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
void JetAnalyzer_HeavyIons::beginJob() {
  std::cout<<"inside the begin job function"<<endl;
}


//------------------------------------------------------------------------------
// endJob
//------------------------------------------------------------------------------
void JetAnalyzer_HeavyIons::endJob()
{
  if (!mOutputFile.empty() && &*edm::Service<DQMStore>())
    {
      edm::Service<DQMStore>()->save(mOutputFile);
    }
}


//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetAnalyzer_HeavyIons::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  // Get the primary vertices
  //----------------------------------------------------------------------------
  edm::Handle<vector<reco::Vertex> > pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);
  reco::Vertex::Point vtx(0,0,0);
  edm::Handle<reco::VertexCollection> vtxs;
  //vtx = getVtx(mEvent);

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

  edm::Handle<reco::Centrality> cent;
  

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
  mEvent.getByToken(centralityToken_, cent);

  mHF->Fill(cent->EtHFtowerSum());
  Float_t HF_energy = cent->EtHFtowerSum();

  const reco::PFCandidateCollection *pfCandidateColl = pfCandidates.product();
  
  Float_t vsPt=0;
  Float_t vsPtInitial = 0;
  Float_t vsArea = 0;
  Int_t NPFpart = 0;
  Int_t NCaloTower = 0;
  Float_t pfPt = 0;
  Float_t pfEta = 0;
  Float_t pfPhi = 0;
  Float_t caloPt = 0;
  Float_t caloEta = 0;
  Float_t caloPhi = 0;
  Float_t SumPt_value = 0;
  Float_t vn_value[200];
  Float_t psin_value[200];

  Float_t vn[fourierOrder_][etaBins_];
  Float_t psin[fourierOrder_][etaBins_];
  Float_t sumpT[etaBins_];

  double edge_pseudorapidity[etaBins_ +1] = {-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191 };
  
  UEParameters vnUE(vn_.product(),fourierOrder_,etaBins_);
  const std::vector<float>& vue = vnUE.get_raw();
  
  for(int ieta = 0;ieta<etaBins_;++ieta){
    sumpT[ieta] = vnUE.get_sum_pt(ieta);
    for(int ifour = 0;ifour<fourierOrder_; ++ifour){
      vn[ifour][ieta] = vnUE.get_vn(ifour,ieta);
      vn_value[ifour * etaBins_ + ieta]= vnUE.get_vn(ifour,ieta);
      mvn->Fill(vn_value[ifour * etaBins_ + ieta]);

      psin[ifour][ieta] = vnUE.get_psin(ifour,ieta);
      psin_value[ifour * etaBins_ + ieta] = vnUE.get_psin(ifour,ieta);
      mpsin->Fill(psin_value[ifour * etaBins_ + ieta]);

      if(0>1)std::cout<<vn[ifour][ieta]<<" "<<psin[ifour][ieta]<<" "<<sumpT[ieta]<<std::endl;

    }
  }

  
  //lets start making the necessary plots 
  Float_t Vs_0_x_minus = sumpT[0]*vn[0][0]*TMath::Cos(0*psin[0][0]);
  Float_t Vs_0_x_plus = sumpT[14]*vn[0][14]*TMath::Cos(0*psin[0][14]);
  Float_t Vs_0_y_minus = sumpT[0]*vn[0][0]*TMath::Sin(0*psin[0][0]);
  Float_t Vs_0_y_plus = sumpT[14]*vn[0][14]*TMath::Sin(0*psin[0][14]);
  Float_t Vs_0_x = Vs_0_x_minus + Vs_0_x_plus;
  Float_t Vs_0_y = Vs_0_y_minus + Vs_0_y_plus;
  
  Float_t Vs_1_x_minus = sumpT[0]*vn[1][0]*TMath::Cos(1*psin[1][0]);
  Float_t Vs_1_x_plus = sumpT[14]*vn[1][14]*TMath::Cos(1*psin[1][14]);
  Float_t Vs_1_y_minus = sumpT[0]*vn[1][0]*TMath::Sin(1*psin[1][0]);
  Float_t Vs_1_y_plus = sumpT[14]*vn[1][14]*TMath::Sin(1*psin[1][14]);
  Float_t Vs_1_x = Vs_1_x_minus + Vs_1_x_plus;
  Float_t Vs_1_y = Vs_1_y_minus + Vs_1_y_plus;
  
  Float_t Vs_2_x_minus = sumpT[0]*vn[2][0]*TMath::Cos(2*psin[2][0]);
  Float_t Vs_2_x_plus = sumpT[14]*vn[2][14]*TMath::Cos(2*psin[2][14]);
  Float_t Vs_2_y_minus = sumpT[0]*vn[2][0]*TMath::Sin(2*psin[2][0]);
  Float_t Vs_2_y_plus = sumpT[14]*vn[2][14]*TMath::Sin(2*psin[2][14]);
  Float_t Vs_2_x = Vs_2_x_minus + Vs_2_x_plus;
  Float_t Vs_2_y = Vs_2_y_minus + Vs_2_y_plus;
  
  // Float_t Vs_3_x_minus = sumpT[0]*vn[3][0]*TMath::Cos(3*psin[3][0]);
  // Float_t Vs_3_x_plus = sumpT[14]*vn[3][14]*TMath::Cos(3*psin[3][14]);
  // Float_t Vs_3_y_minus = sumpT[0]*vn[3][0]*TMath::Sin(3*psin[3][0]);
  // Float_t Vs_3_y_plus = sumpT[14]*vn[3][14]*TMath::Sin(3*psin[3][14]);
  // Float_t Vs_3_x = Vs_3_x_minus + Vs_3_x_plus;
  // Float_t Vs_3_y = Vs_3_y_minus + Vs_3_y_plus;

  // Float_t Vs_4_x_minus = sumpT[0]*vn[4][0]*TMath::Cos(4*psin[4][0]);
  // Float_t Vs_4_x_plus = sumpT[14]*vn[4][14]*TMath::Cos(4*psin[4][14]);
  // Float_t Vs_4_y_minus = sumpT[0]*vn[4][0]*TMath::Sin(4*psin[4][0]);
  // Float_t Vs_4_y_plus = sumpT[14]*vn[4][14]*TMath::Sin(4*psin[4][14]);
  // Float_t Vs_4_x = Vs_4_x_minus + Vs_4_x_plus;
  // Float_t Vs_4_y = Vs_4_y_minus + Vs_4_y_plus;

  mVs_0_x->Fill(Vs_0_x);
  mVs_0_y->Fill(Vs_0_y);
  mVs_1_x->Fill(Vs_1_x);
  mVs_1_y->Fill(Vs_1_y);
  mVs_2_x->Fill(Vs_2_x);
  mVs_2_y->Fill(Vs_2_y);

  mVs_0_x_versus_HF->Fill(HF_energy,Vs_0_x);
  mVs_0_y_versus_HF->Fill(HF_energy,Vs_0_y);
  mVs_1_x_versus_HF->Fill(HF_energy,Vs_1_x);
  mVs_1_y_versus_HF->Fill(HF_energy,Vs_1_y);
  mVs_2_x_versus_HF->Fill(HF_energy,Vs_2_x);
  mVs_2_y_versus_HF->Fill(HF_energy,Vs_2_y);

  Float_t DeltapT = 0;

  if(isCaloJet){

    Float_t SumCaloVsPtInitial[etaBins_];
    Float_t SumCaloVsPt[etaBins_];
    Float_t SumCaloPt[etaBins_];

    Float_t SumSquaredCaloVsPtInitial[etaBins_];
    Float_t SumSquaredCaloVsPt[etaBins_];
    Float_t SumSquaredCaloPt[etaBins_];

    for(int i = 0;i<etaBins_;i++){

      SumCaloVsPtInitial[i] = 0;
      SumCaloVsPt[i] = 0;
      SumCaloPt[i] = 0;
      SumSquaredCaloVsPtInitial[i] = 0;
      SumSquaredCaloVsPt[i] = 0;
      SumSquaredCaloPt[i] = 0;
      
    }
    
    for(unsigned icand = 0;icand<caloCandidates->size(); icand++){

      const CaloTower & tower  = (*caloCandidates)[icand];
      reco::CandidateViewRef ref(calocandidates_,icand);
      //10 is tower pT min
      //if(getEt(caloCandidates.id(),caloCandidates.energy())<10) continue;
      if(tower.p4(vtx).Et() < 0.1) continue;

      DeltapT = 0;
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

      mCaloVsPtInitial_eta_phi->Fill(caloEta,caloPhi,vsPtInitial);
      mCaloVsPt_eta_phi->Fill(caloEta,caloPhi,vsPt);
      mCaloPt_eta_phi->Fill(caloEta,caloPhi,caloPt);

      DeltapT = caloPt - vsPtInitial;
      mDeltapT->Fill(DeltapT);
      mDeltapT_eta->Fill(caloEta,DeltapT);
      mDeltapT_eta_phi->Fill(caloEta,caloPhi,DeltapT);

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
      //mCaloVsPtEqualized
      mCaloArea->Fill(vsArea);
      
    }// calo tower candidate  loop

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
    
      //mSumCaloVsPtInitial->Fill(SumCaloVsPtInitial[k]);
      Evt_SumCaloVsPtInitial = Evt_SumCaloVsPtInitial + SumCaloVsPtInitial[k];
      //mSumCaloVsPt->Fill(SumCaloVsPt[k]);
      Evt_SumCaloVsPt = Evt_SumCaloVsPt + SumCaloVsPt[k];
      //mSumCaloPt->Fill(SumCaloPt[k]);
      Evt_SumCaloPt = Evt_SumCaloPt + SumCaloPt[k];

      mSumCaloVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumCaloVsPtInitial[k]);
      mSumCaloVsPt_eta->Fill(edge_pseudorapidity[k],SumCaloVsPt[k]);
      mSumCaloPt_eta->Fill(edge_pseudorapidity[k],SumCaloPt[k]);

            
      //mSumSquaredCaloVsPtInitial->Fill(SumSquaredCaloVsPtInitial[k]);
      Evt_SumSquaredCaloVsPtInitial = Evt_SumSquaredCaloVsPtInitial + SumSquaredCaloVsPtInitial[k];
      //mSumSquaredCaloVsPt->Fill(SumSquaredCaloVsPt[k]);
      Evt_SumSquaredCaloVsPt = Evt_SumSquaredCaloVsPt + SumSquaredCaloVsPt[k];
      //mSumSquaredCaloPt->Fill(SumSquaredCaloPt[k]);
      Evt_SumSquaredCaloPt = Evt_SumSquaredCaloPt + SumSquaredCaloPt[k];

      mSumSquaredCaloVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumSquaredCaloVsPtInitial[k]);
      mSumSquaredCaloVsPt_eta->Fill(edge_pseudorapidity[k],SumSquaredCaloVsPt[k]);
      mSumSquaredCaloPt_eta->Fill(edge_pseudorapidity[k],SumSquaredCaloPt[k]);
    
    
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

    for(int i = 0;i<etaBins_;i++){

      SumPFVsPtInitial[i] = 0;
      SumPFVsPt[i] = 0;
      SumPFPt[i] = 0;
      SumSquaredPFVsPtInitial[i] = 0;
      SumSquaredPFVsPt[i] = 0;
      SumSquaredPFPt[i] = 0;
      
    }
    
    for(unsigned icand=0;icand<pfCandidateColl->size(); icand++){
    
      const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
      reco::CandidateViewRef ref(pfcandidates_,icand);

      if(pfCandidate.pt() < 5) continue;
      
      if(std::string("Vs")==UEAlgo) {
      
	const reco::VoronoiBackground& voronoi = (*VsBackgrounds)[ref];
	vsPt = voronoi.pt();
	vsPtInitial = voronoi.pt_subtracted();
	vsArea = voronoi.area();
      
      }
    
      NPFpart++;
      pfPt = pfCandidate.pt();
      pfEta = pfCandidate.eta();
      pfPhi = pfCandidate.phi();

      mPFVsPtInitial_eta_phi->Fill(pfEta,pfPhi,vsPtInitial);
      mPFVsPt_eta_phi->Fill(pfEta,pfPhi,vsPt);
      mPFPt_eta_phi->Fill(pfEta,pfPhi,pfPt);

      DeltapT = pfPt - vsPtInitial;
      mDeltapT->Fill(DeltapT);
      mDeltapT_eta->Fill(pfEta,DeltapT);
      mDeltapT_eta_phi->Fill(pfEta,pfPhi,DeltapT);


      for(size_t k = 0;k<nedge_pseudorapidity-1; k++){
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
      mPFVsPtInitial->Fill(vsPtInitial);
      //mPFVsPtEqualized
      mPFArea->Fill(vsArea);
    
    }// pf candidate loop 

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
    
      //mSumPFVsPtInitial->Fill(SumPFVsPtInitial[k]);
      Evt_SumPFVsPtInitial = Evt_SumPFVsPtInitial + SumPFVsPtInitial[k];
      //mSumPFVsPt->Fill(SumPFVsPt[k]);
      Evt_SumPFVsPt = Evt_SumPFVsPt + SumPFVsPt[k];
      //mSumPFPt->Fill(SumPFPt[k]);
      Evt_SumPFPt = Evt_SumPFPt + SumPFPt[k];

      mSumPFVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumPFVsPtInitial[k]);
      mSumPFVsPt_eta->Fill(edge_pseudorapidity[k],SumPFVsPt[k]);
      mSumPFPt_eta->Fill(edge_pseudorapidity[k],SumPFPt[k]);

      //mSumSquaredPFVsPtInitial->Fill(SumSquaredPFVsPtInitial[k]);
      Evt_SumSquaredPFVsPtInitial = Evt_SumSquaredPFVsPtInitial + SumSquaredPFVsPtInitial[k];
      //mSumSquaredPFVsPt->Fill(SumSquaredPFVsPt[k]);
      Evt_SumSquaredPFVsPt = Evt_SumSquaredPFVsPt + SumSquaredPFVsPt[k];
      //mSumSquaredPFPt->Fill(SumSquaredPFPt[k]);
      Evt_SumSquaredPFPt = Evt_SumSquaredPFPt + SumSquaredPFPt[k];

      mSumSquaredPFVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumSquaredPFVsPtInitial[k]);
      mSumSquaredPFVsPt_eta->Fill(edge_pseudorapidity[k],SumSquaredPFVsPt[k]);
      mSumSquaredPFPt_eta->Fill(edge_pseudorapidity[k],SumSquaredPFPt[k]);

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
      for (unsigned ijet=0; ijet<caloJets->size(); ijet++) recoJets.push_back((*caloJets)[ijet]);
    }
  
  if (isJPTJet)
    {
      for (unsigned ijet=0; ijet<jptJets->size(); ijet++) recoJets.push_back((*jptJets)[ijet]);
    }
  
  if (isPFJet) {
    if(std::string("Pu")==UEAlgo){
      for (unsigned ijet=0; ijet<basicJets->size();ijet++) recoJets.push_back((*basicJets)[ijet]);
    }
    if(std::string("Vs")==UEAlgo){
      for (unsigned ijet=0; ijet<pfJets->size(); ijet++) recoJets.push_back((*pfJets)[ijet]);
    }
  }
  

    
  if (isCaloJet && !caloJets.isValid()) return;
  if (isJPTJet  && !jptJets.isValid())  return;
  if (isPFJet){
    if(std::string("Pu")==UEAlgo){if(!basicJets.isValid())   return;}
    if(std::string("Vs")==UEAlgo){if(!pfJets.isValid())   return;}
  }
  

  // int nJet_E_20_40 = 0;
  // int nJet_B_20_40 = 0;
  // int nJet_E_40 = 0;
  // int nJet_B_40 = 0;
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
      
    }
  }
  
  if (mNJets_40) mNJets_40->Fill(nJet_40); 
  
}


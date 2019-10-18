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

JetAnalyzer_HeavyIons::JetAnalyzer_HeavyIons(const edm::ParameterSet &iConfig)
    : mInputCollection(iConfig.getParameter<edm::InputTag>("src")),
      mInputVtxCollection(iConfig.getUntrackedParameter<edm::InputTag>("srcVtx", edm::InputTag("hiSelectedVertex"))),
      mInputPFCandCollection(iConfig.getParameter<edm::InputTag>("PFcands")),
      mInputCsCandCollection(iConfig.exists("CScands") ? iConfig.getParameter<edm::InputTag>("CScands")
                                                       : edm::InputTag()),
      mOutputFile(iConfig.getUntrackedParameter<std::string>("OutputFile", "")),
      JetType(iConfig.getUntrackedParameter<std::string>("JetType")),
      UEAlgo(iConfig.getUntrackedParameter<std::string>("UEAlgo")),
      mRecoJetPtThreshold(iConfig.getParameter<double>("recoJetPtThreshold")),
      mReverseEnergyFractionThreshold(iConfig.getParameter<double>("reverseEnergyFractionThreshold")),
      mRThreshold(iConfig.getParameter<double>("RThreshold")),
      JetCorrectionService(iConfig.getParameter<std::string>("JetCorrections")) {
  std::string inputCollectionLabel(mInputCollection.label());

  isCaloJet = (std::string("calo") == JetType);
  isJPTJet = (std::string("jpt") == JetType);
  isPFJet = (std::string("pf") == JetType);

  //consumes
  pvToken_ = consumes<std::vector<reco::Vertex>>(edm::InputTag("offlinePrimaryVertices"));
  caloTowersToken_ = consumes<CaloTowerCollection>(edm::InputTag("towerMaker"));
  if (isCaloJet)
    caloJetsToken_ = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isJPTJet)
    jptJetsToken_ = consumes<reco::JPTJetCollection>(mInputCollection);
  if (isPFJet) {
    if (std::string("Pu") == UEAlgo)
      basicJetsToken_ = consumes<reco::BasicJetCollection>(mInputCollection);
    if (std::string("Cs") == UEAlgo)
      pfJetsToken_ = consumes<reco::PFJetCollection>(mInputCollection);
  }

  pfCandToken_ = consumes<reco::PFCandidateCollection>(mInputPFCandCollection);
  csCandToken_ = mayConsume<reco::PFCandidateCollection>(mInputCsCandCollection);
  pfCandViewToken_ = consumes<reco::CandidateView>(mInputPFCandCollection);
  caloCandViewToken_ = consumes<reco::CandidateView>(edm::InputTag("towerMaker"));

  centralityTag_ = iConfig.getParameter<InputTag>("centralitycollection");
  centralityToken = consumes<reco::Centrality>(centralityTag_);

  centralityBinToken = mayConsume<int>(iConfig.exists("centralitybincollection")
                                           ? iConfig.getParameter<edm::InputTag>("centralitybincollection")
                                           : edm::InputTag());

  hiVertexToken_ = consumes<std::vector<reco::Vertex>>(mInputVtxCollection);

  etaToken_ = mayConsume<std::vector<double>>(iConfig.exists("etaMap") ? iConfig.getParameter<edm::InputTag>("etaMap")
                                                                       : edm::InputTag());
  rhoToken_ = mayConsume<std::vector<double>>(iConfig.exists("rho") ? iConfig.getParameter<edm::InputTag>("rho")
                                                                    : edm::InputTag());
  rhomToken_ = mayConsume<std::vector<double>>(iConfig.exists("rhom") ? iConfig.getParameter<edm::InputTag>("rhom")
                                                                      : edm::InputTag());

  // need to initialize the PF cand histograms : which are also event variables
  if (isPFJet) {
    mNPFpart = nullptr;
    mPFPt = nullptr;
    mPFEta = nullptr;
    mPFPhi = nullptr;

    mSumPFPt = nullptr;
    mSumPFPt_eta = nullptr;
    mSumSquaredPFPt = nullptr;
    mSumSquaredPFPt_eta = nullptr;
    mSumPFPt_HF = nullptr;

    mPFDeltaR = nullptr;
    mPFDeltaR_Scaled_R = nullptr;

    for (int ieta = 0; ieta < etaBins_; ieta++) {
      mSumPFPtEtaDep[ieta] = nullptr;
    }

    //cs-specific histograms
    mRhoDist_vsEta = nullptr;
    mRhoMDist_vsEta = nullptr;
    mRhoDist_vsPt = nullptr;
    mRhoMDist_vsPt = nullptr;

    rhoEtaRange = nullptr;
    for (int ieta = 0; ieta < etaBins_; ieta++) {
      mCSCandpT_vsPt[ieta] = nullptr;
      mRhoDist_vsCent[ieta] = nullptr;
      mRhoMDist_vsCent[ieta] = nullptr;
      for (int ipt = 0; ipt < ptBins_; ipt++) {
        mSubtractedEFrac[ipt][ieta] = nullptr;
        mSubtractedE[ipt][ieta] = nullptr;
      }
    }

    mPFCandpT_vs_eta_Unknown = nullptr;        // pf id 0
    mPFCandpT_vs_eta_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_vs_eta_electron = nullptr;       // pf id - 2
    mPFCandpT_vs_eta_muon = nullptr;           // pf id - 3
    mPFCandpT_vs_eta_photon = nullptr;         // pf id - 4
    mPFCandpT_vs_eta_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_vs_eta_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_vs_eta_EME_inHF = nullptr;       // pf id - 7

    mPFCandpT_Barrel_Unknown = nullptr;        // pf id 0
    mPFCandpT_Barrel_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_Barrel_electron = nullptr;       // pf id - 2
    mPFCandpT_Barrel_muon = nullptr;           // pf id - 3
    mPFCandpT_Barrel_photon = nullptr;         // pf id - 4
    mPFCandpT_Barrel_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_Barrel_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_Barrel_EME_inHF = nullptr;       // pf id - 7

    mPFCandpT_Endcap_Unknown = nullptr;        // pf id 0
    mPFCandpT_Endcap_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_Endcap_electron = nullptr;       // pf id - 2
    mPFCandpT_Endcap_muon = nullptr;           // pf id - 3
    mPFCandpT_Endcap_photon = nullptr;         // pf id - 4
    mPFCandpT_Endcap_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_Endcap_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_Endcap_EME_inHF = nullptr;       // pf id - 7

    mPFCandpT_Forward_Unknown = nullptr;        // pf id 0
    mPFCandpT_Forward_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_Forward_electron = nullptr;       // pf id - 2
    mPFCandpT_Forward_muon = nullptr;           // pf id - 3
    mPFCandpT_Forward_photon = nullptr;         // pf id - 4
    mPFCandpT_Forward_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_Forward_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_Forward_EME_inHF = nullptr;       // pf id - 7
  }
  if (isCaloJet) {
    mNCalopart = nullptr;
    mCaloPt = nullptr;
    mCaloEta = nullptr;
    mCaloPhi = nullptr;

    mSumCaloPt = nullptr;
    mSumCaloPt_eta = nullptr;
    mSumSquaredCaloPt = nullptr;
    mSumSquaredCaloPt_eta = nullptr;
    mSumCaloPt_HF = nullptr;

    for (int ieta = 0; ieta < etaBins_; ieta++) {
      mSumCaloPtEtaDep[ieta] = nullptr;
    }
  }

  mSumpt = nullptr;

  // Events variables
  mNvtx = nullptr;
  mHF = nullptr;

  // added Jan 12th 2015

  // Jet parameters
  mEta = nullptr;
  mPhi = nullptr;
  mEnergy = nullptr;
  mP = nullptr;
  mPt = nullptr;
  mMass = nullptr;
  mConstituents = nullptr;
  mJetArea = nullptr;
  mjetpileup = nullptr;
  mNJets_40 = nullptr;
  mNJets = nullptr;
}

void JetAnalyzer_HeavyIons::bookHistograms(DQMStore::IBooker &ibooker,
                                           edm::Run const &iRun,
                                           edm::EventSetup const &iSetup) {
  ibooker.setCurrentFolder("JetMET/HIJetValidation/" + mInputCollection.label());

  TH2F *h2D_etabins_vs_pt2 =
      new TH2F("h2D_etabins_vs_pt2", ";#eta;sum p_{T}^{2}", etaBins_, edge_pseudorapidity, 10000, 0, 10000);
  TH2F *h2D_etabins_vs_pt =
      new TH2F("h2D_etabins_vs_pt", ";#eta;sum p_{T}", etaBins_, edge_pseudorapidity, 500, 0, 500);
  TH2F *h2D_pfcand_etabins_vs_pt =
      new TH2F("h2D_etabins_vs_pt", ";#eta;sum p_{T}", etaBins_, edge_pseudorapidity, 300, 0, 300);

  const int nHihfBins = 100;
  const double hihfBins[nHihfBins + 1] = {
      0,           11.282305,   11.82962,    12.344717,   13.029054,   13.698554,   14.36821,    15.140326,
      15.845786,   16.684441,   17.449186,   18.364939,   19.247023,   20.448898,   21.776642,   22.870239,
      24.405788,   26.366919,   28.340206,   30.661842,   33.657627,   36.656773,   40.028049,   44.274784,
      48.583706,   52.981358,   56.860199,   61.559853,   66.663689,   72.768196,   78.265915,   84.744431,
      92.483459,   100.281021,  108.646576,  117.023911,  125.901093,  135.224899,  147.046875,  159.864258,
      171.06015,   184.76535,   197.687103,  212.873535,  229.276413,  245.175369,  262.498322,  280.54599,
      299.570801,  317.188446,  336.99881,   357.960144,  374.725922,  400.638367,  426.062103,  453.07251,
      483.99704,   517.556396,  549.421143,  578.050781,  608.358643,  640.940979,  680.361755,  719.215027,
      757.798645,  793.882385,  839.83728,   887.268127,  931.233276,  980.856689,  1023.191833, 1080.281494,
      1138.363892, 1191.303345, 1251.439453, 1305.288818, 1368.290894, 1433.700684, 1501.597412, 1557.918335,
      1625.636475, 1695.08374,  1761.771484, 1848.941162, 1938.178345, 2027.55603,  2127.364014, 2226.186523,
      2315.188965, 2399.225342, 2501.608643, 2611.077881, 2726.316162, 2848.74707,  2972.975342, 3096.565674,
      3219.530762, 3361.178223, 3568.028564, 3765.690186, 50000};

  TH2F *h2D_etabins_forRho =
      new TH2F("etabinsForRho", "#rho vs. #eta;#eta;#rho", etaBins_, edge_pseudorapidity, 500, 0, 300);
  TH2F *h2D_ptBins_forRho = new TH2F("ptBinsForRho", "#rho vs. p_{T};p_{T};#rho", 300, 0, 300, 500, 0, 300);
  TH2F *h2D_centBins_forRho = new TH2F("centBinsForRho", "dummy;HIHF;#rho", nHihfBins, hihfBins, 500, 0, 300);

  TH2F *h2D_etabins_forRhoM =
      new TH2F("etabinsForRho", "#rho_{M} vs. #eta;#eta;#rho_{M}", etaBins_, edge_pseudorapidity, 100, 0, 1.5);
  TH2F *h2D_ptBins_forRhoM = new TH2F("ptBinsForRho", "#rho_{M} vs. p_{T};p_{T};#rho_{M}", 300, 0, 300, 100, 0, 1.5);
  TH2F *h2D_centBins_forRhoM = new TH2F("centBinsForRho", "dummy;HIHF;#rho_{M}", nHihfBins, hihfBins, 100, 0, 1.5);

  if (isPFJet) {
    mNPFpart = ibooker.book1D("NPFpart", "No of particle flow candidates", 1000, 0, 1000);
    mPFPt = ibooker.book1D("PFPt", "PF candidate p_{T}", 10000, -500, 500);
    mPFEta = ibooker.book1D("PFEta", "PF candidate #eta", 120, -6, 6);
    mPFPhi = ibooker.book1D("PFPhi", "PF candidate #phi", 70, -3.5, 3.5);

    mPFDeltaR = ibooker.book1D("PFDeltaR", "PF candidate DeltaR", 100, 0, 4);  //MZ
    mPFDeltaR_Scaled_R =
        ibooker.book1D("PFDeltaR_Scaled_R", "PF candidate DeltaR Divided by DeltaR square", 100, 0, 4);  //MZ

    mSumPFPt = ibooker.book1D("SumPFPt", "Sum of initial PF p_{T}", 1000, -10000, 10000);
    mSumPFPt_eta = ibooker.book2D("SumPFPt_etaBins", h2D_etabins_vs_pt);

    mSumSquaredPFPt = ibooker.book1D("SumSquaredPFPt", "Sum of initial PF p_{T} squared", 10000, 0, 10000);
    mSumSquaredPFPt_eta = ibooker.book2D("SumSquaredPFPt_etaBins", h2D_etabins_vs_pt2);

    mSumPFPt_HF = ibooker.book2D(
        "SumPFPt_HF", "HF energy (y axis) vs Sum initial PF p_{T} (x axis)", 1000, -1000, 1000, 1000, 0, 10000);

    for (int ieta = 0; ieta < etaBins_; ieta++) {
      int range = 1000;
      if (ieta < 2 || etaBins_ - ieta <= 2)
        range = 500;
      const char *lc = edge_pseudorapidity[ieta] < 0 ? "n" : "p";
      const char *rc = edge_pseudorapidity[ieta + 1] < 0 ? "n" : "p";
      std::string histoName =
          Form("mSumCaloPt_%s%.3g_%s%.3g", lc, abs(edge_pseudorapidity[ieta]), rc, abs(edge_pseudorapidity[ieta + 1]));
      for (int id = 0; id < 2; id++) {
        if (histoName.find(".") != std::string::npos) {
          histoName.replace(histoName.find("."), 1, "p");
        }
      }
      mSumPFPtEtaDep[ieta] = ibooker.book1D(
          histoName.c_str(),
          Form("Sum PFPt in the eta range %.3g to %.3g", edge_pseudorapidity[ieta], edge_pseudorapidity[ieta + 1]),
          500,
          0,
          range);
    }

    if (std::string("Cs") == UEAlgo) {
      mRhoDist_vsEta = ibooker.book2D("rhoDist_vsEta", h2D_etabins_forRho);
      mRhoMDist_vsEta = ibooker.book2D("rhoMDist_vsEta", h2D_etabins_forRhoM);
      mRhoDist_vsPt = ibooker.book2D("rhoDist_vsPt", h2D_ptBins_forRho);
      mRhoMDist_vsPt = ibooker.book2D("rhoMDist_vsPt", h2D_ptBins_forRhoM);

      //this is kind of a janky way to fill the eta since i can't get it from the edm::Event here... - kjung
      rhoEtaRange = ibooker.book1D("rhoEtaRange", "", 500, -5.5, 5.5);
      for (int ieta = 0; ieta < etaBins_; ieta++) {
        mCSCandpT_vsPt[ieta] =
            ibooker.book1D(Form("csCandPt_etaBin%d", ieta), "CS candidate pt, eta-by-eta", 150, 0, 300);

        const char *lc = edge_pseudorapidity[ieta] < 0 ? "n" : "p";
        const char *rc = edge_pseudorapidity[ieta + 1] < 0 ? "n" : "p";
        std::string histoName = Form(
            "Dist_vsCent_%s%.3g_%s%.3g", lc, abs(edge_pseudorapidity[ieta]), rc, abs(edge_pseudorapidity[ieta + 1]));
        for (int id = 0; id < 2; id++) {
          if (histoName.find(".") != std::string::npos) {
            histoName.replace(histoName.find("."), 1, "p");
          }
        }
        std::string rhoName = "rho";
        rhoName.append(histoName);
        h2D_centBins_forRho->SetTitle(Form(
            "#rho vs. HIHF in the range %.3g < #eta < %.3g", edge_pseudorapidity[ieta], edge_pseudorapidity[ieta + 1]));
        mRhoDist_vsCent[ieta] = ibooker.book2D(rhoName.c_str(), h2D_centBins_forRho);
        std::string rhoMName = "rhoM";
        rhoMName.append(histoName);
        h2D_centBins_forRhoM->SetTitle(Form("#rho_{M} vs. HIHF in the range %.3g < #eta < %.3g",
                                            edge_pseudorapidity[ieta],
                                            edge_pseudorapidity[ieta + 1]));
        mRhoMDist_vsCent[ieta] = ibooker.book2D(rhoMName.c_str(), h2D_centBins_forRhoM);
        for (int ipt = 0; ipt < ptBins_; ipt++) {
          mSubtractedEFrac[ipt][ieta] =
              ibooker.book1D(Form("subtractedEFrac_JetPt%d_to_%d_etaBin%d", ptBin[ipt], ptBin[ipt + 1], ieta),
                             "subtracted fraction of CS jet",
                             50,
                             0,
                             1);
          mSubtractedE[ipt][ieta] =
              ibooker.book1D(Form("subtractedE_JetPt%d_to_%d_etaBin%d", ptBin[ipt], ptBin[ipt + 1], ieta),
                             "subtracted total of CS jet",
                             300,
                             0,
                             300);
        }
        mCSCand_corrPFcand[ieta] = ibooker.book2D(
            Form("csCandCorrPF%d", ieta), "CS to PF candidate correlation, eta-by-eta", 300, 0, 300, 300, 0, 300);
      }
    }

    mPFCandpT_vs_eta_Unknown = ibooker.book2D("PF_cand_X_unknown", h2D_pfcand_etabins_vs_pt);         // pf id 0
    mPFCandpT_vs_eta_ChargedHadron = ibooker.book2D("PF_cand_chargedHad", h2D_pfcand_etabins_vs_pt);  // pf id - 1
    mPFCandpT_vs_eta_electron = ibooker.book2D("PF_cand_electron", h2D_pfcand_etabins_vs_pt);         // pf id - 2
    mPFCandpT_vs_eta_muon = ibooker.book2D("PF_cand_muon", h2D_pfcand_etabins_vs_pt);                 // pf id - 3
    mPFCandpT_vs_eta_photon = ibooker.book2D("PF_cand_photon", h2D_pfcand_etabins_vs_pt);             // pf id - 4
    mPFCandpT_vs_eta_NeutralHadron = ibooker.book2D("PF_cand_neutralHad", h2D_pfcand_etabins_vs_pt);  // pf id - 5
    mPFCandpT_vs_eta_HadE_inHF = ibooker.book2D("PF_cand_HadEner_inHF", h2D_pfcand_etabins_vs_pt);    // pf id - 6
    mPFCandpT_vs_eta_EME_inHF = ibooker.book2D("PF_cand_EMEner_inHF", h2D_pfcand_etabins_vs_pt);      // pf id - 7

    mPFCandpT_Barrel_Unknown = ibooker.book1D("mPFCandpT_Barrel_Unknown",
                                              Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                              300,
                                              0,
                                              300);  // pf id  - 0
    mPFCandpT_Barrel_ChargedHadron = ibooker.book1D("mPFCandpT_Barrel_ChargedHadron",
                                                    Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                                    300,
                                                    0,
                                                    300);  // pf id - 1
    mPFCandpT_Barrel_electron = ibooker.book1D("mPFCandpT_Barrel_electron",
                                               Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                               300,
                                               0,
                                               300);  // pf id - 2
    mPFCandpT_Barrel_muon = ibooker.book1D("mPFCandpT_Barrel_muon",
                                           Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                           300,
                                           0,
                                           300);  // pf id - 3
    mPFCandpT_Barrel_photon = ibooker.book1D("mPFCandpT_Barrel_photon",
                                             Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                             300,
                                             0,
                                             300);  // pf id - 4
    mPFCandpT_Barrel_NeutralHadron = ibooker.book1D("mPFCandpT_Barrel_NeutralHadron",
                                                    Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                                    300,
                                                    0,
                                                    300);  // pf id - 5
    mPFCandpT_Barrel_HadE_inHF = ibooker.book1D("mPFCandpT_Barrel_HadE_inHF",
                                                Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                                300,
                                                0,
                                                300);  // pf id - 6
    mPFCandpT_Barrel_EME_inHF = ibooker.book1D("mPFCandpT_Barrel_EME_inHF",
                                               Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                               300,
                                               0,
                                               300);  // pf id - 7

    mPFCandpT_Endcap_Unknown =
        ibooker.book1D("mPFCandpT_Endcap_Unknown",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 0
    mPFCandpT_Endcap_ChargedHadron =
        ibooker.book1D("mPFCandpT_Endcap_ChargedHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 1
    mPFCandpT_Endcap_electron =
        ibooker.book1D("mPFCandpT_Endcap_electron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 2
    mPFCandpT_Endcap_muon =
        ibooker.book1D("mPFCandpT_Endcap_muon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 3
    mPFCandpT_Endcap_photon =
        ibooker.book1D("mPFCandpT_Endcap_photon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 4
    mPFCandpT_Endcap_NeutralHadron =
        ibooker.book1D("mPFCandpT_Endcap_NeutralHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 5
    mPFCandpT_Endcap_HadE_inHF =
        ibooker.book1D("mPFCandpT_Endcap_HadE_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 6
    mPFCandpT_Endcap_EME_inHF =
        ibooker.book1D("mPFCandpT_Endcap_EME_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 7

    mPFCandpT_Forward_Unknown =
        ibooker.book1D("mPFCandpT_Forward_Unknown",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 0
    mPFCandpT_Forward_ChargedHadron =
        ibooker.book1D("mPFCandpT_Forward_ChargedHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 1
    mPFCandpT_Forward_electron =
        ibooker.book1D("mPFCandpT_Forward_electron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 2
    mPFCandpT_Forward_muon =
        ibooker.book1D("mPFCandpT_Forward_muon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 3
    mPFCandpT_Forward_photon =
        ibooker.book1D("mPFCandpT_Forward_photon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 4
    mPFCandpT_Forward_NeutralHadron =
        ibooker.book1D("mPFCandpT_Forward_NeutralHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 5
    mPFCandpT_Forward_HadE_inHF =
        ibooker.book1D("mPFCandpT_Forward_HadE_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 6
    mPFCandpT_Forward_EME_inHF =
        ibooker.book1D("mPFCandpT_Forward_EME_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 7
  }

  if (isCaloJet) {
    mNCalopart = ibooker.book1D("NCalopart", "No of particle flow candidates", 1000, 0, 10000);
    mCaloPt = ibooker.book1D("CaloPt", "Calo candidate p_{T}", 1000, -5000, 5000);
    mCaloEta = ibooker.book1D("CaloEta", "Calo candidate #eta", 120, -6, 6);
    mCaloPhi = ibooker.book1D("CaloPhi", "Calo candidate #phi", 70, -3.5, 3.5);

    mSumCaloPt = ibooker.book1D("SumCaloPt", "Sum Calo p_{T}", 1000, -10000, 10000);
    mSumCaloPt_eta = ibooker.book2D("SumCaloPt_etaBins", h2D_etabins_vs_pt);

    mSumSquaredCaloPt = ibooker.book1D("SumSquaredCaloPt", "Sum of initial Calo tower p_{T} squared", 10000, 0, 10000);
    mSumSquaredCaloPt_eta = ibooker.book2D("SumSquaredCaloPt_etaBins", h2D_etabins_vs_pt2);

    mSumCaloPt_HF =
        ibooker.book2D("SumCaloPt_HF", "HF Energy (y axis) vs Sum Calo tower p_{T}", 1000, -1000, 1000, 1000, 0, 10000);

    for (int ieta = 0; ieta < etaBins_; ieta++) {
      int range = 1000;
      if (ieta < 2 || etaBins_ - ieta <= 2)
        range = 5000;
      const char *lc = edge_pseudorapidity[ieta] < 0 ? "n" : "p";
      const char *rc = edge_pseudorapidity[ieta + 1] < 0 ? "n" : "p";
      std::string histoName =
          Form("mSumCaloPt_%s%.3g_%s%.3g", lc, abs(edge_pseudorapidity[ieta]), rc, abs(edge_pseudorapidity[ieta + 1]));
      for (int id = 0; id < 2; id++) {
        if (histoName.find(".") != std::string::npos) {
          histoName.replace(histoName.find("."), 1, "p");
        }
      }
      mSumCaloPtEtaDep[ieta] = ibooker.book1D(histoName.c_str(),
                                              Form("Sum Calo tower Pt in the eta range %.3g to %.3g",
                                                   edge_pseudorapidity[ieta],
                                                   edge_pseudorapidity[ieta + 1]),
                                              1000,
                                              -1 * range,
                                              range);
    }
  }

  // particle flow variables histograms
  mSumpt = ibooker.book1D("SumpT", "Sum p_{T} of all the PF candidates per event", 1000, 0, 10000);

  // Event variables
  mNvtx = ibooker.book1D("Nvtx", "number of vertices", 60, 0, 60);
  mHF = ibooker.book1D("HF", "HF energy distribution", 1000, 0, 10000);

  // Jet parameters
  mEta = ibooker.book1D("Eta", "Eta", 120, -6, 6);
  mPhi = ibooker.book1D("Phi", "Phi", 70, -3.5, 3.5);
  mPt = ibooker.book1D("Pt", "Pt", 1000, 0, 500);
  mP = ibooker.book1D("P", "P", 100, 0, 1000);
  mEnergy = ibooker.book1D("Energy", "Energy", 100, 0, 1000);
  mMass = ibooker.book1D("Mass", "Mass", 100, 0, 200);
  mConstituents = ibooker.book1D("Constituents", "Constituents", 100, 0, 100);
  mJetArea = ibooker.book1D("JetArea", "JetArea", 100, 0, 4);
  mjetpileup = ibooker.book1D("jetPileUp", "jetPileUp", 100, 0, 150);
  mNJets_40 = ibooker.book1D("NJets_pt_greater_40", "NJets pT > 40 GeV", 50, 0, 50);
  mNJets = ibooker.book1D("NJets", "NJets", 100, 0, 100);

  if (mOutputFile.empty())
    LogInfo("OutputInfo") << " Histograms will NOT be saved";
  else
    LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;

  delete h2D_etabins_vs_pt2;
  delete h2D_etabins_vs_pt;
  delete h2D_pfcand_etabins_vs_pt;
  delete h2D_etabins_forRho;
  delete h2D_ptBins_forRho;
  delete h2D_centBins_forRho;
  delete h2D_centBins_forRhoM;
}

//------------------------------------------------------------------------------
// ~JetAnalyzer_HeavyIons
//------------------------------------------------------------------------------
JetAnalyzer_HeavyIons::~JetAnalyzer_HeavyIons() {}

//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
//void JetAnalyzer_HeavyIons::beginJob() {
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
void JetAnalyzer_HeavyIons::analyze(const edm::Event &mEvent, const edm::EventSetup &mSetup) {
  // switch(mEvent.id().event() == 15296770)
  // case 1:
  //   break;

  // Get the primary vertices
  edm::Handle<vector<reco::Vertex>> pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);
  reco::Vertex::Point vtx(0, 0, 0);
  edm::Handle<reco::VertexCollection> vtxs;

  mEvent.getByToken(hiVertexToken_, vtxs);
  int greatestvtx = 0;
  int nVertex = vtxs->size();

  for (unsigned int i = 0; i < vtxs->size(); ++i) {
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if (daughter > (*vtxs)[greatestvtx].tracksSize())
      greatestvtx = i;
  }

  if (nVertex <= 0) {
    vtx = reco::Vertex::Point(0, 0, 0);
  }
  vtx = (*vtxs)[greatestvtx].position();

  int nGoodVertices = 0;

  if (pvHandle.isValid()) {
    for (unsigned i = 0; i < pvHandle->size(); i++) {
      if ((*pvHandle)[i].ndof() > 4 && (fabs((*pvHandle)[i].z()) <= 24) && (fabs((*pvHandle)[i].position().rho()) <= 2))
        nGoodVertices++;
    }
  }

  mNvtx->Fill(nGoodVertices);

  // Get the Jet collection
  //----------------------------------------------------------------------------

  std::vector<Jet> recoJets;
  recoJets.clear();

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;
  edm::Handle<BasicJetCollection> basicJets;

  // Get the Particle flow candidates and the Voronoi variables
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  edm::Handle<reco::PFCandidateCollection> csCandidates;
  edm::Handle<CaloTowerCollection> caloCandidates;
  edm::Handle<reco::CandidateView> pfcandidates_;
  edm::Handle<reco::CandidateView> calocandidates_;

  //Get the new CS stuff
  edm::Handle<std::vector<double>> etaRanges;
  edm::Handle<std::vector<double>> rho;
  edm::Handle<std::vector<double>> rhom;
  if (std::string("Cs") == UEAlgo) {
    mEvent.getByToken(etaToken_, etaRanges);
    mEvent.getByToken(rhoToken_, rho);
    mEvent.getByToken(rhomToken_, rhom);
    const int rhoSize = (int)etaRanges->size();
    double rhoRange[rhoSize];
    for (int irho = 0; irho < rhoSize; irho++) {
      rhoRange[irho] = etaRanges->at(irho);
    }
    double yaxisLimits[501];
    for (int ibin = 0; ibin < 501; ibin++)
      yaxisLimits[ibin] = ibin * 2;
    if (mRhoDist_vsEta->getNbinsX() != rhoSize - 1) {
      mRhoDist_vsEta->getTH2F()->SetBins(
          rhoSize - 1, const_cast<double *>(rhoRange), 500, const_cast<double *>(yaxisLimits));
      mRhoMDist_vsEta->getTH2F()->SetBins(
          rhoSize - 1, const_cast<double *>(rhoRange), 500, const_cast<double *>(yaxisLimits));
    }
  }

  if (isCaloJet)
    mEvent.getByToken(caloJetsToken_, caloJets);
  if (isJPTJet)
    mEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet) {
    if (std::string("Pu") == UEAlgo)
      mEvent.getByToken(basicJetsToken_, basicJets);
    if (std::string("Cs") == UEAlgo)
      mEvent.getByToken(pfJetsToken_, pfJets);
    if (std::string("Vs") == UEAlgo)
      return;  //avoid running Vs jets
  }

  mEvent.getByToken(pfCandToken_, pfCandidates);
  mEvent.getByToken(pfCandViewToken_, pfcandidates_);
  if (std::string("Cs") == UEAlgo)
    mEvent.getByToken(csCandToken_, csCandidates);

  mEvent.getByToken(caloTowersToken_, caloCandidates);
  mEvent.getByToken(caloCandViewToken_, calocandidates_);

  // get the centrality
  edm::Handle<reco::Centrality> cent;
  mEvent.getByToken(centralityToken, cent);  //_centralitytag comes from the cfg

  mHF->Fill(cent->EtHFtowerSum());
  Float_t HF_energy = cent->EtHFtowerSum();

  //for later when centrality gets added to RelVal
  //edm::Handle<int> cbin;
  //mEvent.getByToken(centralityBinToken, cbin);

  if (!cent.isValid())
    return;

  /*int hibin = -999;
  if(cbin.isValid()){
    hibin = *cbin;
  }*/

  const reco::PFCandidateCollection *pfCandidateColl = pfCandidates.product();

  Int_t NPFpart = 0;
  Int_t NCaloTower = 0;
  Float_t pfPt = 0;
  Float_t pfEta = 0;
  Float_t pfPhi = 0;
  Int_t pfID = 0;
  Float_t pfDeltaR = 0;
  Float_t caloPt = 0;
  Float_t caloEta = 0;
  Float_t caloPhi = 0;
  Float_t SumPt_value = 0;

  vector<vector<float>> numbers;
  vector<float> tempVector;
  numbers.clear();
  tempVector.clear();

  if (isCaloJet) {
    Float_t SumCaloPt[etaBins_];
    Float_t SumSquaredCaloPt[etaBins_];

    // Need to set up histograms to get the RMS values for each pT bin
    TH1F *hSumCaloPt[nedge_pseudorapidity - 1];

    for (int i = 0; i < etaBins_; ++i) {
      SumCaloPt[i] = 0;
      SumSquaredCaloPt[i] = 0;
      hSumCaloPt[i] = new TH1F(Form("hSumCaloPt_%d", i), "", 10000, -10000, 10000);
    }

    for (unsigned icand = 0; icand < caloCandidates->size(); icand++) {
      const CaloTower &tower = (*caloCandidates)[icand];
      reco::CandidateViewRef ref(calocandidates_, icand);
      //10 is tower pT min
      if (tower.p4(vtx).Et() < 0.1)
        continue;

      NCaloTower++;

      caloPt = tower.p4(vtx).Et();
      caloEta = tower.p4(vtx).Eta();
      caloPhi = tower.p4(vtx).Phi();

      for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
        if (caloEta >= edge_pseudorapidity[k] && caloEta < edge_pseudorapidity[k + 1]) {
          SumCaloPt[k] = SumCaloPt[k] + caloPt;
          SumSquaredCaloPt[k] = SumSquaredCaloPt[k] + caloPt * caloPt;
          break;
        }  // eta selection statement

      }  // eta bin loop

      SumPt_value = SumPt_value + caloPt;

      mCaloPt->Fill(caloPt);
      mCaloEta->Fill(caloEta);
      mCaloPhi->Fill(caloPhi);

    }  // calo tower candidate  loop

    for (int k = 0; k < nedge_pseudorapidity - 1; k++) {
      hSumCaloPt[k]->Fill(SumCaloPt[k]);

    }  // eta bin loop

    Float_t Evt_SumCaloPt = 0;
    Float_t Evt_SumSquaredCaloPt = 0;

    for (int ieta = 0; ieta < etaBins_; ieta++) {
      mSumCaloPtEtaDep[ieta]->Fill(SumCaloPt[ieta]);
    }

    for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
      Evt_SumCaloPt = Evt_SumCaloPt + SumCaloPt[k];
      mSumCaloPt_eta->Fill(edge_pseudorapidity[k], SumCaloPt[k]);

      Evt_SumSquaredCaloPt = Evt_SumSquaredCaloPt + SumSquaredCaloPt[k];
      mSumSquaredCaloPt_eta->Fill(edge_pseudorapidity[k], hSumCaloPt[k]->GetRMS(1));

      delete hSumCaloPt[k];

    }  // eta bin loop

    mSumCaloPt->Fill(Evt_SumCaloPt);
    mSumCaloPt_HF->Fill(Evt_SumCaloPt, HF_energy);

    mSumSquaredCaloPt->Fill(Evt_SumSquaredCaloPt);

    mNCalopart->Fill(NCaloTower);
    mSumpt->Fill(SumPt_value);

  }  // is calo jet

  if (isPFJet) {
    Float_t SumPFPt[etaBins_];

    Float_t SumSquaredPFPt[etaBins_];

    // Need to set up histograms to get the RMS values for each pT bin
    TH1F *hSumPFPt[nedge_pseudorapidity - 1];

    for (int i = 0; i < etaBins_; i++) {
      SumPFPt[i] = 0;
      SumSquaredPFPt[i] = 0;

      hSumPFPt[i] = new TH1F(Form("hSumPFPt_%d", i), "", 10000, -10000, 10000);
    }

    vector<vector<float>> PF_Space(1, vector<float>(3));

    if (std::string("Cs") == UEAlgo) {
      const reco::PFCandidateCollection *csCandidateColl = csCandidates.product();

      for (unsigned iCScand = 0; iCScand < csCandidateColl->size(); iCScand++) {
        assert(csCandidateColl->size() <= pfCandidateColl->size());
        const reco::PFCandidate csCandidate = csCandidateColl->at(iCScand);
        const reco::PFCandidate pfCandidate = pfCandidateColl->at(iCScand);
        int ieta = 0;
        while (csCandidate.eta() > edge_pseudorapidity[ieta] && ieta < etaBins_ - 1)
          ieta++;
        mCSCandpT_vsPt[ieta]->Fill(csCandidate.pt());
        mCSCand_corrPFcand[ieta]->Fill(csCandidate.pt(), pfCandidate.pt());
      }
    }

    for (unsigned icand = 0; icand < pfCandidateColl->size(); icand++) {
      const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
      reco::CandidateViewRef ref(pfcandidates_, icand);
      if (pfCandidate.pt() < 5)
        continue;

      NPFpart++;
      pfPt = pfCandidate.pt();
      pfEta = pfCandidate.eta();
      pfPhi = pfCandidate.phi();
      pfID = pfCandidate.particleId();

      bool isBarrel = false;
      bool isEndcap = false;
      bool isForward = false;

      if (fabs(pfEta) < BarrelEta)
        isBarrel = true;
      if (fabs(pfEta) >= BarrelEta && fabs(pfEta) < EndcapEta)
        isEndcap = true;
      if (fabs(pfEta) >= EndcapEta && fabs(pfEta) < ForwardEta)
        isForward = true;

      switch (pfID) {
        case 0:
          mPFCandpT_vs_eta_Unknown->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_Unknown->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_Unknown->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_Unknown->Fill(pfPt);
          break;
        case 1:
          mPFCandpT_vs_eta_ChargedHadron->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_ChargedHadron->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_ChargedHadron->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_ChargedHadron->Fill(pfPt);
          break;
        case 2:
          mPFCandpT_vs_eta_electron->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_electron->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_electron->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_electron->Fill(pfPt);
          break;
        case 3:
          mPFCandpT_vs_eta_muon->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_muon->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_muon->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_muon->Fill(pfPt);
          break;
        case 4:
          mPFCandpT_vs_eta_photon->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_photon->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_photon->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_photon->Fill(pfPt);
          break;
        case 5:
          mPFCandpT_vs_eta_NeutralHadron->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_NeutralHadron->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_NeutralHadron->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_NeutralHadron->Fill(pfPt);
          break;
        case 6:
          mPFCandpT_vs_eta_HadE_inHF->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_HadE_inHF->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_HadE_inHF->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_HadE_inHF->Fill(pfPt);
          break;
        case 7:
          mPFCandpT_vs_eta_EME_inHF->Fill(pfPt, pfEta);
          if (isBarrel)
            mPFCandpT_Barrel_EME_inHF->Fill(pfPt);
          if (isEndcap)
            mPFCandpT_Endcap_EME_inHF->Fill(pfPt);
          if (isForward)
            mPFCandpT_Forward_EME_inHF->Fill(pfPt);
          break;
      }

      //Fill 2d vector matrix
      tempVector.push_back(pfPt);
      tempVector.push_back(pfEta);
      tempVector.push_back(pfPhi);

      numbers.push_back(tempVector);
      tempVector.clear();

      for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
        if (pfEta >= edge_pseudorapidity[k] && pfEta < edge_pseudorapidity[k + 1]) {
          SumPFPt[k] = SumPFPt[k] + pfPt;
          SumSquaredPFPt[k] = SumSquaredPFPt[k] + pfPt * pfPt;
          break;
        }  // eta selection statement

      }  // eta bin loop

      SumPt_value = SumPt_value + pfPt;

      mPFPt->Fill(pfPt);
      mPFEta->Fill(pfEta);
      mPFPhi->Fill(pfPhi);

    }  // pf candidate loop

    for (int k = 0; k < nedge_pseudorapidity - 1; k++) {
      hSumPFPt[k]->Fill(SumPFPt[k]);

    }  // eta bin loop

    Float_t Evt_SumPFPt = 0;
    Float_t Evt_SumSquaredPFPt = 0;

    for (int ieta = 0; ieta < etaBins_; ieta++) {
      mSumPFPtEtaDep[ieta]->Fill(SumPFPt[ieta]);
    }

    for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
      Evt_SumPFPt = Evt_SumPFPt + SumPFPt[k];
      mSumPFPt_eta->Fill(edge_pseudorapidity[k], SumPFPt[k]);

      Evt_SumSquaredPFPt = Evt_SumSquaredPFPt + SumSquaredPFPt[k];
      mSumSquaredPFPt_eta->Fill(edge_pseudorapidity[k], hSumPFPt[k]->GetRMS(1));

      delete hSumPFPt[k];

    }  // eta bin loop

    mSumPFPt->Fill(Evt_SumPFPt);
    mSumPFPt_HF->Fill(Evt_SumPFPt, HF_energy);

    mSumSquaredPFPt->Fill(Evt_SumSquaredPFPt);

    mNPFpart->Fill(NPFpart);
    mSumpt->Fill(SumPt_value);
  }

  if (isCaloJet) {
    for (unsigned ijet = 0; ijet < caloJets->size(); ijet++) {
      recoJets.push_back((*caloJets)[ijet]);
    }
  }

  if (isJPTJet) {
    for (unsigned ijet = 0; ijet < jptJets->size(); ijet++)
      recoJets.push_back((*jptJets)[ijet]);
  }

  if (isPFJet) {
    if (std::string("Pu") == UEAlgo) {
      for (unsigned ijet = 0; ijet < basicJets->size(); ijet++) {
        recoJets.push_back((*basicJets)[ijet]);
      }
    }
    if (std::string("Cs") == UEAlgo) {
      for (unsigned ijet = 0; ijet < pfJets->size(); ijet++) {
        recoJets.push_back((*pfJets)[ijet]);
      }
    }
  }

  if (isCaloJet && !caloJets.isValid()) {
    return;
  }
  if (isJPTJet && !jptJets.isValid()) {
    return;
  }
  if (isPFJet) {
    if (std::string("Pu") == UEAlgo) {
      if (!basicJets.isValid())
        return;
    }
    if (std::string("Cs") == UEAlgo) {
      if (!pfJets.isValid())
        return;
    }
    if (std::string("Vs") == UEAlgo) {
      return;
    }
  }

  int nJet_40 = 0;

  mNJets->Fill(recoJets.size());

  for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {
    if (recoJets[ijet].pt() > mRecoJetPtThreshold) {
      //counting forward and barrel jets
      // get an idea of no of jets with pT>40 GeV
      if (recoJets[ijet].pt() > 40)
        nJet_40++;
      if (mEta)
        mEta->Fill(recoJets[ijet].eta());
      if (mjetpileup)
        mjetpileup->Fill(recoJets[ijet].pileup());
      if (mJetArea)
        mJetArea->Fill(recoJets[ijet].jetArea());
      if (mPhi)
        mPhi->Fill(recoJets[ijet].phi());
      if (mEnergy)
        mEnergy->Fill(recoJets[ijet].energy());
      if (mP)
        mP->Fill(recoJets[ijet].p());
      if (mPt)
        mPt->Fill(recoJets[ijet].pt());
      if (mMass)
        mMass->Fill(recoJets[ijet].mass());
      if (mConstituents)
        mConstituents->Fill(recoJets[ijet].nConstituents());

      if (std::string("Cs") == UEAlgo) {
        int ipt = 0, ieta = 0;
        while (recoJets[ijet].pt() > ptBin[ipt + 1] && ipt < ptBins_ - 1)
          ipt++;
        while (recoJets[ijet].eta() > etaRanges->at(ieta + 1) && ieta < (int)(rho->size() - 1))
          ieta++;
        mSubtractedEFrac[ipt][ieta]->Fill((double)recoJets[ijet].pileup() / (double)recoJets[ijet].energy());
        mSubtractedE[ipt][ieta]->Fill(recoJets[ijet].pileup());

        for (unsigned irho = 0; irho < rho->size(); irho++) {
          mRhoDist_vsEta->Fill(recoJets[ijet].eta(), rho->at(irho));
          mRhoMDist_vsEta->Fill(recoJets[ijet].eta(), rhom->at(irho));
          mRhoDist_vsPt->Fill(recoJets[ijet].pt(), rho->at(irho));
          mRhoMDist_vsPt->Fill(recoJets[ijet].pt(), rhom->at(irho));
          mRhoDist_vsCent[ieta]->Fill(HF_energy, rho->at(irho));
          mRhoMDist_vsCent[ieta]->Fill(HF_energy, rhom->at(irho));
        }
      }

      for (size_t iii = 0; iii < numbers.size(); iii++) {
        pfDeltaR = sqrt((numbers[iii][2] - recoJets[ijet].phi()) * (numbers[iii][2] - recoJets[ijet].phi()) +
                        (numbers[iii][1] - recoJets[ijet].eta()) * (numbers[iii][1] - recoJets[ijet].eta()));  //MZ

        mPFDeltaR->Fill(pfDeltaR);                                  //MZ
        mPFDeltaR_Scaled_R->Fill(pfDeltaR, 1. / pow(pfDeltaR, 2));  //MZ
      }
    }
  }
  if (mNJets_40)
    mNJets_40->Fill(nJet_40);

  numbers.clear();
}

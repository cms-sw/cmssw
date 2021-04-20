#include <iostream>
#include <iomanip>
//

#include "DQMOffline/EGamma/plugins/PhotonAnalyzer.h"

/** \class PhotonAnalyzer
 **
 **
 **  $Id: PhotonAnalyzer
 **  authors:
 **   Nancy Marinelli, U. of Notre Dame, US
 **   Jamie Antonelli, U. of Notre Dame, US
 **
 ***/

using namespace std;

PhotonAnalyzer::PhotonAnalyzer(const edm::ParameterSet& pset) {
  fName_ = pset.getParameter<string>("analyzerName");
  prescaleFactor_ = pset.getUntrackedParameter<int>("prescaleFactor", 1);

  photon_token_ = consumes<vector<reco::Photon> >(pset.getParameter<edm::InputTag>("phoProducer"));
  barrelRecHit_token_ = consumes<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit> > >(
      pset.getParameter<edm::InputTag>("barrelRecHitProducer"));
  PhotonIDLoose_token_ = consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("photonIDLoose"));
  PhotonIDTight_token_ = consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("photonIDTight"));
  endcapRecHit_token_ = consumes<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit> > >(
      pset.getParameter<edm::InputTag>("endcapRecHitProducer"));
  triggerEvent_token_ = consumes<trigger::TriggerEvent>(pset.getParameter<edm::InputTag>("triggerEvent"));
  offline_pvToken_ = consumes<reco::VertexCollection>(
      pset.getUntrackedParameter<edm::InputTag>("offlinePV", edm::InputTag("offlinePrimaryVertices")));

  minPhoEtCut_ = pset.getParameter<double>("minPhoEtCut");
  photonMaxEta_ = pset.getParameter<double>("maxPhoEta");
  invMassEtCut_ = pset.getParameter<double>("invMassEtCut");
  cutStep_ = pset.getParameter<double>("cutStep");
  numberOfSteps_ = pset.getParameter<int>("numberOfSteps");
  useBinning_ = pset.getParameter<bool>("useBinning");
  useTriggerFiltering_ = pset.getParameter<bool>("useTriggerFiltering");
  minimalSetOfHistos_ = pset.getParameter<bool>("minimalSetOfHistos");
  excludeBkgHistos_ = pset.getParameter<bool>("excludeBkgHistos");
  standAlone_ = pset.getParameter<bool>("standAlone");
  isolationStrength_ = pset.getParameter<int>("isolationStrength");
  isHeavyIon_ = pset.getUntrackedParameter<bool>("isHeavyIon", false);

  histo_index_photons_ = 0;
  histo_index_conversions_ = 0;
  histo_index_efficiency_ = 0;
  histo_index_invMass_ = 0;

  nEvt_ = 0;

  // Determining parts...
  parts_.push_back("AllEcal");
  parts_.push_back("Barrel");
  parts_.push_back("Endcaps");
  // ...and types
  types_.push_back("All");
  types_.push_back("GoodCandidate");
  if (!excludeBkgHistos_) {
    types_.push_back("Background");
  }

  // Histogram parameters
  etaBin_ = pset.getParameter<int>("etaBin");
  etaMin_ = pset.getParameter<double>("etaMin");
  etaMax_ = pset.getParameter<double>("etaMax");

  etBin_ = pset.getParameter<int>("etBin");
  etMin_ = pset.getParameter<double>("etMin");
  etMax_ = pset.getParameter<double>("etMax");

  phiBin_ = pset.getParameter<int>("phiBin");
  phiMin_ = pset.getParameter<double>("phiMin");
  phiMax_ = pset.getParameter<double>("phiMax");

  eBin_ = pset.getParameter<int>("eBin");
  eMin_ = pset.getParameter<double>("eMin");
  eMax_ = pset.getParameter<double>("eMax");

  numberBin_ = pset.getParameter<int>("numberBin");
  numberMin_ = pset.getParameter<double>("numberMin");
  numberMax_ = pset.getParameter<double>("numberMax");

  r9Bin_ = pset.getParameter<int>("r9Bin");
  r9Min_ = pset.getParameter<double>("r9Min");
  r9Max_ = pset.getParameter<double>("r9Max");

  sigmaIetaBin_ = pset.getParameter<int>("sigmaIetaBin");
  sigmaIetaMin_ = pset.getParameter<double>("sigmaIetaMin");
  sigmaIetaMax_ = pset.getParameter<double>("sigmaIetaMax");

  sumBin_ = pset.getParameter<int>("sumBin");
  sumMin_ = pset.getParameter<double>("sumMin");
  sumMax_ = pset.getParameter<double>("sumMax");

  hOverEBin_ = pset.getParameter<int>("hOverEBin");
  hOverEMin_ = pset.getParameter<double>("hOverEMin");
  hOverEMax_ = pset.getParameter<double>("hOverEMax");

  eOverPBin_ = pset.getParameter<int>("eOverPBin");
  eOverPMin_ = pset.getParameter<double>("eOverPMin");
  eOverPMax_ = pset.getParameter<double>("eOverPMax");

  dPhiTracksBin_ = pset.getParameter<int>("dPhiTracksBin");
  dPhiTracksMin_ = pset.getParameter<double>("dPhiTracksMin");
  dPhiTracksMax_ = pset.getParameter<double>("dPhiTracksMax");

  dEtaTracksBin_ = pset.getParameter<int>("dEtaTracksBin");
  dEtaTracksMin_ = pset.getParameter<double>("dEtaTracksMin");
  dEtaTracksMax_ = pset.getParameter<double>("dEtaTracksMax");

  chi2Bin_ = pset.getParameter<int>("chi2Bin");
  chi2Min_ = pset.getParameter<double>("chi2Min");
  chi2Max_ = pset.getParameter<double>("chi2Max");

  zBin_ = pset.getParameter<int>("zBin");
  zMin_ = pset.getParameter<double>("zMin");
  zMax_ = pset.getParameter<double>("zMax");

  rBin_ = pset.getParameter<int>("rBin");
  rMin_ = pset.getParameter<double>("rMin");
  rMax_ = pset.getParameter<double>("rMax");

  xBin_ = pset.getParameter<int>("xBin");
  xMin_ = pset.getParameter<double>("xMin");
  xMax_ = pset.getParameter<double>("xMax");

  yBin_ = pset.getParameter<int>("yBin");
  yMin_ = pset.getParameter<double>("yMin");
  yMax_ = pset.getParameter<double>("yMax");

  reducedEtBin_ = etBin_ / 4;
  reducedEtaBin_ = etaBin_ / 4;
  reducedR9Bin_ = r9Bin_ / 4;
  reducedSumBin_ = sumBin_ / 4;
}

PhotonAnalyzer::~PhotonAnalyzer() {}

void PhotonAnalyzer::bookHistograms(DQMStore::IBooker& iBooker,
                                    edm::Run const& /* iRun */,
                                    edm::EventSetup const& /* iSetup */) {
  bookHistogramsForHistogramCounts(iBooker);

  bookHistogramsEfficiency(iBooker);
  bookHistogramsInvMass(iBooker);
  bookHistogramsPhotons(iBooker);
  bookHistogramsConversions(iBooker);

  fillHistogramsForHistogramCounts(iBooker);
}

void PhotonAnalyzer::bookHistogramsForHistogramCounts(DQMStore::IBooker& iBooker) {
  iBooker.setCurrentFolder("Egamma/" + fName_ + "/");
  // Int values stored in MEs to keep track of how many histograms are in each folder
  totalNumberOfHistos_efficiencyFolder = iBooker.bookInt("numberOfHistogramsInEfficiencyFolder");
  totalNumberOfHistos_invMassFolder = iBooker.bookInt("numberOfHistogramsInInvMassFolder");
  totalNumberOfHistos_photonsFolder = iBooker.bookInt("numberOfHistogramsInPhotonsFolder");
  totalNumberOfHistos_conversionsFolder = iBooker.bookInt("numberOfHistogramsInConversionsFolder");
}

void PhotonAnalyzer::fillHistogramsForHistogramCounts(DQMStore::IBooker& iBooker) {
  iBooker.setCurrentFolder("Egamma/" + fName_ + "/");
  totalNumberOfHistos_efficiencyFolder->Fill(histo_index_efficiency_);
  totalNumberOfHistos_invMassFolder->Fill(histo_index_invMass_);
  totalNumberOfHistos_photonsFolder->Fill(histo_index_photons_);
  totalNumberOfHistos_conversionsFolder->Fill(histo_index_conversions_);
}

void PhotonAnalyzer::bookHistogramsEfficiency(DQMStore::IBooker& iBooker) {
  // Set folder
  iBooker.setCurrentFolder("Egamma/" + fName_ + "/Efficiencies");

  // Don't number these histograms with the "bookHisto" method, since they'll be erased in the offline client
  h_phoEta_Loose_ = iBooker.book1D("phoEtaLoose", "Loose Photon #eta", etaBin_, etaMin_, etaMax_);
  h_phoEta_Tight_ = iBooker.book1D("phoEtaTight", "Tight Photon #eta", etaBin_, etaMin_, etaMax_);

  h_phoEt_Loose_ = iBooker.book1D("phoEtLoose", "Loose Photon E_{T}", etBin_, etMin_, etMax_);
  h_phoEt_Tight_ = iBooker.book1D("phoEtTight", "Tight Photon E_{T}", etBin_, etMin_, etMax_);

  h_phoEta_preHLT_ = iBooker.book1D("phoEtaPreHLT", "Photon #eta: before HLT", etaBin_, etaMin_, etaMax_);
  h_phoEta_postHLT_ = iBooker.book1D("phoEtaPostHLT", "Photon #eta: after HLT", etaBin_, etaMin_, etaMax_);
  h_phoEt_preHLT_ = iBooker.book1D("phoEtPreHLT", "Photon E_{T}: before HLT", etBin_, etMin_, etMax_);
  h_phoEt_postHLT_ = iBooker.book1D("phoEtPostHLT", "Photon E_{T}: after HLT", etBin_, etMin_, etMax_);

  h_convEta_Loose_ = iBooker.book1D("convEtaLoose", "Converted Loose Photon #eta", etaBin_, etaMin_, etaMax_);
  h_convEta_Tight_ = iBooker.book1D("convEtaTight", "Converted Tight Photon #eta", etaBin_, etaMin_, etaMax_);
  h_convEt_Loose_ = iBooker.book1D("convEtLoose", "Converted Loose Photon E_{T}", etBin_, etMin_, etMax_);
  h_convEt_Tight_ = iBooker.book1D("convEtTight", "Converted Tight Photon E_{T}", etBin_, etMin_, etMax_);

  h_phoEta_Vertex_ =
      iBooker.book1D("phoEtaVertex", "Converted Photons before valid vertex cut: #eta", etaBin_, etaMin_, etaMax_);

  // Some temporary vectors
  vector<MonitorElement*> temp1DVectorEta;
  vector<MonitorElement*> temp1DVectorPhi;
  vector<vector<MonitorElement*> > temp2DVectorPhi;

  for (int cut = 0; cut != numberOfSteps_; ++cut) {       //looping over Et cut values
    for (uint type = 0; type != types_.size(); ++type) {  //looping over isolation type
      currentFolder_.str("");
      currentFolder_ << "Egamma/" + fName_ + "/" << types_[type] << "Photons/Et above " << (cut + 1) * cutStep_
                     << " GeV/Conversions";
      iBooker.setCurrentFolder(currentFolder_.str());

      temp1DVectorEta.push_back(
          iBooker.book1D("phoConvEtaForEfficiency", "Converted Photon #eta;#eta", etaBin_, etaMin_, etaMax_));
      for (uint part = 0; part != parts_.size(); ++part) {
        temp1DVectorPhi.push_back(iBooker.book1D(
            "phoConvPhiForEfficiency" + parts_[part], "Converted Photon #phi;#phi", phiBin_, phiMin_, phiMax_));
      }
      temp2DVectorPhi.push_back(temp1DVectorPhi);
      temp1DVectorPhi.clear();
    }
    h_phoConvEtaForEfficiency_.push_back(temp1DVectorEta);
    temp1DVectorEta.clear();
    h_phoConvPhiForEfficiency_.push_back(temp2DVectorPhi);
    temp2DVectorPhi.clear();
  }
}

void PhotonAnalyzer::bookHistogramsInvMass(DQMStore::IBooker& iBooker) {
  // Set folder
  iBooker.setCurrentFolder("Egamma/" + fName_ + "/InvMass");

  h_invMassAllPhotons_ = bookHisto(iBooker,
                                   "invMassAllIsolatedPhotons",
                                   "Two photon invariant mass: All isolated photons;M (GeV)",
                                   etBin_,
                                   etMin_,
                                   etMax_);
  h_invMassPhotonsEBarrel_ = bookHisto(iBooker,
                                       "invMassIsoPhotonsEBarrel",
                                       "Two photon invariant mass: isolated photons in barrel; M (GeV)",
                                       etBin_,
                                       etMin_,
                                       etMax_);
  h_invMassPhotonsEEndcap_ = bookHisto(iBooker,
                                       "invMassIsoPhotonsEEndcap",
                                       "Two photon invariant mass: isolated photons in endcap; M (GeV)",
                                       etBin_,
                                       etMin_,
                                       etMax_);
  h_invMassPhotonsEEndcapEBarrel_ = bookHisto(iBooker,
                                              "invMassIsoPhotonsEEndcapEBarrel",
                                              "Two photon invariant mass: isolated photons in endcap-barrel; M (GeV)",
                                              etBin_,
                                              etMin_,
                                              etMax_);

  h_invMassZeroWithTracks_ = bookHisto(
      iBooker, "invMassZeroWithTracks", "Two photon invariant mass: Neither has tracks;M (GeV)", etBin_, etMin_, etMax_);
  h_invMassOneWithTracks_ = bookHisto(
      iBooker, "invMassOneWithTracks", "Two photon invariant mass: Only one has tracks;M (GeV)", etBin_, etMin_, etMax_);
  h_invMassTwoWithTracks_ = bookHisto(
      iBooker, "invMassTwoWithTracks", "Two photon invariant mass: Both have tracks;M (GeV)", etBin_, etMin_, etMax_);

  h_nRecoVtx_ = bookHisto(iBooker, "nOfflineVtx", "# of Offline Vertices", 200, -0.5, 199.5);
}

void PhotonAnalyzer::bookHistogramsPhotons(DQMStore::IBooker& iBooker) {
  // Set folder
  // Folder is set by the book2DHistoVector and book3DHistoVector methods

  //ENERGY VARIABLES
  book3DHistoVector(iBooker, h_phoE_, "1D", "phoE", "Energy;E (GeV)", eBin_, eMin_, eMax_);
  book3DHistoVector(iBooker, h_phoSigmaEoverE_, "1D", "phoSigmaEoverE", "#sigma_{E}/E; #sigma_{E}/E", 100, 0., 0.08);
  book3DHistoVector(iBooker,
                    p_phoSigmaEoverEvsNVtx_,
                    "Profile",
                    "phoSigmaEoverEvsNVtx",
                    "#sigma_{E}/E vs NVtx; N_{vtx}; #sigma_{E}/E",
                    200,
                    -0.5,
                    199.5,
                    100,
                    0.,
                    0.08);
  book3DHistoVector(iBooker, h_phoEt_, "1D", "phoEt", "E_{T};E_{T} (GeV)", etBin_, etMin_, etMax_);

  //NUMBER OF PHOTONS
  book3DHistoVector(
      iBooker, h_nPho_, "1D", "nPho", "Number of Photons per Event;# #gamma", numberBin_, numberMin_, numberMax_);

  //GEOMETRICAL VARIABLES
  //photon eta/phi
  book2DHistoVector(iBooker, h_phoEta_, "1D", "phoEta", "#eta;#eta", etaBin_, etaMin_, etaMax_);
  book3DHistoVector(iBooker, h_phoPhi_, "1D", "phoPhi", "#phi;#phi", phiBin_, phiMin_, phiMax_);

  //supercluster eta/phi
  book2DHistoVector(iBooker, h_scEta_, "1D", "scEta", "SuperCluster #eta;#eta", etaBin_, etaMin_, etaMax_);
  book3DHistoVector(iBooker, h_scPhi_, "1D", "scPhi", "SuperCluster #phi;#phi", phiBin_, phiMin_, phiMax_);

  //SHOWER SHAPE VARIABLES
  //r9
  book3DHistoVector(iBooker, h_r9_, "1D", "r9", "R9;R9", r9Bin_, r9Min_, r9Max_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_r9VsEt_,
                      "2D",
                      "r9VsEt2D",
                      "R9 vs E_{T};E_{T} (GeV);R9",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedR9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_r9VsEt_,
                    "Profile",
                    "r9VsEt",
                    "Avg R9 vs E_{T};E_{T} (GeV);R9",
                    etBin_,
                    etMin_,
                    etMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_r9VsEta_,
                      "2D",
                      "r9VsEta2D",
                      "R9 vs #eta;#eta;R9",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedR9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_r9VsEta_,
                    "Profile",
                    "r9VsEta",
                    "Avg R9 vs #eta;#eta;R9",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);

  //sigma ieta ieta
  book3DHistoVector(iBooker,
                    h_phoSigmaIetaIeta_,
                    "1D",
                    "phoSigmaIetaIeta",
                    "#sigma_{i#etai#eta};#sigma_{i#etai#eta}",
                    sigmaIetaBin_,
                    sigmaIetaMin_,
                    sigmaIetaMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_sigmaIetaIetaVsEta_,
                      "2D",
                      "sigmaIetaIetaVsEta2D",
                      "#sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      sigmaIetaBin_,
                      sigmaIetaMin_,
                      sigmaIetaMax_);
  }
  book2DHistoVector(iBooker,
                    p_sigmaIetaIetaVsEta_,
                    "Profile",
                    "sigmaIetaIetaVsEta",
                    "Avg #sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    sigmaIetaBin_,
                    sigmaIetaMin_,
                    sigmaIetaMax_);

  //e1x5
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_e1x5VsEt_,
                      "2D",
                      "e1x5VsEt2D",
                      "E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedEtBin_,
                      etMin_,
                      etMax_);
  }
  book2DHistoVector(iBooker,
                    p_e1x5VsEt_,
                    "Profile",
                    "e1x5VsEt",
                    "Avg E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",
                    etBin_,
                    etMin_,
                    etMax_,
                    etBin_,
                    etMin_,
                    etMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_e1x5VsEta_,
                      "2D",
                      "e1x5VsEta2D",
                      "E1x5 vs #eta;#eta;E1X5 (GeV)",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedEtBin_,
                      etMin_,
                      etMax_);
  }
  book2DHistoVector(iBooker,
                    p_e1x5VsEta_,
                    "Profile",
                    "e1x5VsEta",
                    "Avg E1x5 vs #eta;#eta;E1X5 (GeV)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    etBin_,
                    etMin_,
                    etMax_);

  //e2x5
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_e2x5VsEt_,
                      "2D",
                      "e2x5VsEt2D",
                      "E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedEtBin_,
                      etMin_,
                      etMax_);
  }
  book2DHistoVector(iBooker,
                    p_e2x5VsEt_,
                    "Profile",
                    "e2x5VsEt",
                    "Avg E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",
                    etBin_,
                    etMin_,
                    etMax_,
                    etBin_,
                    etMin_,
                    etMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_e2x5VsEta_,
                      "2D",
                      "e2x5VsEta2D",
                      "E2x5 vs #eta;#eta;E2X5 (GeV)",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedEtBin_,
                      etMin_,
                      etMax_);
  }
  book2DHistoVector(iBooker,
                    p_e2x5VsEta_,
                    "Profile",
                    "e2x5VsEta",
                    "Avg E2x5 vs #eta;#eta;E2X5 (GeV)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    etBin_,
                    etMin_,
                    etMax_);

  //r1x5
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_r1x5VsEt_,
                      "2D",
                      "r1x5VsEt2D",
                      "R1x5 vs E_{T};E_{T} (GeV);R1X5",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedR9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_r1x5VsEt_,
                    "Profile",
                    "r1x5VsEt",
                    "Avg R1x5 vs E_{T};E_{T} (GeV);R1X5",
                    etBin_,
                    etMin_,
                    etMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_r1x5VsEta_,
                      "2D",
                      "r1x5VsEta2D",
                      "R1x5 vs #eta;#eta;R1X5",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedR9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_r1x5VsEta_,
                    "Profile",
                    "r1x5VsEta",
                    "Avg R1x5 vs #eta;#eta;R1X5",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);

  //r2x5
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_r2x5VsEt_,
                      "2D",
                      "r2x5VsEt2D",
                      "R2x5 vs E_{T};E_{T} (GeV);R2X5",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedR9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_r2x5VsEt_,
                    "Profile",
                    "r2x5VsEt",
                    "Avg R2x5 vs E_{T};E_{T} (GeV);R2X5",
                    etBin_,
                    etMin_,
                    etMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_r2x5VsEta_,
                      "2D",
                      "r2x5VsEta2D",
                      "R2x5 vs #eta;#eta;R2X5",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedR9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_r2x5VsEta_,
                    "Profile",
                    "r2x5VsEta",
                    "Avg R2x5 vs #eta;#eta;R2X5",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);

  //maxEXtalOver3x3
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_maxEXtalOver3x3VsEt_,
                      "2D",
                      "maxEXtalOver3x3VsEt2D",
                      "(Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      r9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_maxEXtalOver3x3VsEt_,
                    "Profile",
                    "maxEXtalOver3x3VsEt",
                    "Avg (Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",
                    etBin_,
                    etMin_,
                    etMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_maxEXtalOver3x3VsEta_,
                      "2D",
                      "maxEXtalOver3x3VsEta2D",
                      "(Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      r9Bin_,
                      r9Min_,
                      r9Max_);
  }
  book2DHistoVector(iBooker,
                    p_maxEXtalOver3x3VsEta_,
                    "Profile",
                    "maxEXtalOver3x3VsEta",
                    "Avg (Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    r9Bin_,
                    r9Min_,
                    r9Max_);

  //TRACK ISOLATION VARIABLES
  //nTrackIsolSolid
  book2DHistoVector(iBooker,
                    h_nTrackIsolSolid_,
                    "1D",
                    "nIsoTracksSolid",
                    "Number Of Tracks in the Solid Iso Cone;# tracks",
                    numberBin_,
                    numberMin_,
                    numberMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_nTrackIsolSolidVsEt_,
                      "2D",
                      "nIsoTracksSolidVsEt2D",
                      "Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      numberBin_,
                      numberMin_,
                      numberMax_);
  }
  book2DHistoVector(iBooker,
                    p_nTrackIsolSolidVsEt_,
                    "Profile",
                    "nIsoTracksSolidVsEt",
                    "Avg Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",
                    etBin_,
                    etMin_,
                    etMax_,
                    numberBin_,
                    numberMin_,
                    numberMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_nTrackIsolSolidVsEta_,
                      "2D",
                      "nIsoTracksSolidVsEta2D",
                      "Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      numberBin_,
                      numberMin_,
                      numberMax_);
  }
  book2DHistoVector(iBooker,
                    p_nTrackIsolSolidVsEta_,
                    "Profile",
                    "nIsoTracksSolidVsEta",
                    "Avg Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    numberBin_,
                    numberMin_,
                    numberMax_);

  //nTrackIsolHollow
  book2DHistoVector(iBooker,
                    h_nTrackIsolHollow_,
                    "1D",
                    "nIsoTracksHollow",
                    "Number Of Tracks in the Hollow Iso Cone;# tracks",
                    numberBin_,
                    numberMin_,
                    numberMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_nTrackIsolHollowVsEt_,
                      "2D",
                      "nIsoTracksHollowVsEt2D",
                      "Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      numberBin_,
                      numberMin_,
                      numberMax_);
  }
  book2DHistoVector(iBooker,
                    p_nTrackIsolHollowVsEt_,
                    "Profile",
                    "nIsoTracksHollowVsEt",
                    "Avg Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",
                    etBin_,
                    etMin_,
                    etMax_,
                    numberBin_,
                    numberMin_,
                    numberMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_nTrackIsolHollowVsEta_,
                      "2D",
                      "nIsoTracksHollowVsEta2D",
                      "Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      numberBin_,
                      numberMin_,
                      numberMax_);
  }
  book2DHistoVector(iBooker,
                    p_nTrackIsolHollowVsEta_,
                    "Profile",
                    "nIsoTracksHollowVsEta",
                    "Avg Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    numberBin_,
                    numberMin_,
                    numberMax_);

  //trackPtSumSolid
  book2DHistoVector(iBooker,
                    h_trackPtSumSolid_,
                    "1D",
                    "isoPtSumSolid",
                    "Track P_{T} Sum in the Solid Iso Cone;P_{T} (GeV)",
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_trackPtSumSolidVsEt_,
                      "2D",
                      "isoPtSumSolidVsEt2D",
                      "Track P_{T} Sum in the Solid Iso Cone;E_{T} (GeV);P_{T} (GeV)",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book2DHistoVector(iBooker,
                    p_trackPtSumSolidVsEt_,
                    "Profile",
                    "isoPtSumSolidVsEt",
                    "Avg Track P_{T} Sum in the Solid Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",
                    etBin_,
                    etMin_,
                    etMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_trackPtSumSolidVsEta_,
                      "2D",
                      "isoPtSumSolidVsEta2D",
                      "Track P_{T} Sum in the Solid Iso Cone;#eta;P_{T} (GeV)",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book2DHistoVector(iBooker,
                    p_trackPtSumSolidVsEta_,
                    "Profile",
                    "isoPtSumSolidVsEta",
                    "Avg Track P_{T} Sum in the Solid Iso Cone vs #eta;#eta;P_{T} (GeV)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);

  //trackPtSumHollow
  book2DHistoVector(iBooker,
                    h_trackPtSumHollow_,
                    "1D",
                    "isoPtSumHollow",
                    "Track P_{T} Sum in the Hollow Iso Cone;P_{T} (GeV)",
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_trackPtSumHollowVsEt_,
                      "2D",
                      "isoPtSumHollowVsEt2D",
                      "Track P_{T} Sum in the Hollow Iso Cone;E_{T} (GeV);P_{T} (GeV)",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book2DHistoVector(iBooker,
                    p_trackPtSumHollowVsEt_,
                    "Profile",
                    "isoPtSumHollowVsEt",
                    "Avg Track P_{T} Sum in the Hollow Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",
                    etBin_,
                    etMin_,
                    etMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_trackPtSumHollowVsEta_,
                      "2D",
                      "isoPtSumHollowVsEta2D",
                      "Track P_{T} Sum in the Hollow Iso Cone;#eta;P_{T} (GeV)",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book2DHistoVector(iBooker,
                    p_trackPtSumHollowVsEta_,
                    "Profile",
                    "isoPtSumHollowVsEta",
                    "Avg Track P_{T} Sum in the Hollow Iso Cone vs #eta;#eta;P_{T} (GeV)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);

  //CALORIMETER ISOLATION VARIABLES
  //ecal sum
  book2DHistoVector(
      iBooker, h_ecalSum_, "1D", "ecalSum", "Ecal Sum in the Iso Cone;E (GeV)", sumBin_, sumMin_, sumMax_);
  book2DHistoVector(iBooker,
                    h_ecalSumEBarrel_,
                    "1D",
                    "ecalSumEBarrel",
                    "Ecal Sum in the IsoCone for Barrel;E (GeV)",
                    sumBin_,
                    sumMin_,
                    sumMax_);
  book2DHistoVector(iBooker,
                    h_ecalSumEEndcap_,
                    "1D",
                    "ecalSumEEndcap",
                    "Ecal Sum in the IsoCone for Endcap;E (GeV)",
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_ecalSumVsEt_,
                      "2D",
                      "ecalSumVsEt2D",
                      "Ecal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book3DHistoVector(iBooker,
                    p_ecalSumVsEt_,
                    "Profile",
                    "ecalSumVsEt",
                    "Avg Ecal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",
                    etBin_,
                    etMin_,
                    etMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_ecalSumVsEta_,
                      "2D",
                      "ecalSumVsEta2D",
                      "Ecal Sum in the Iso Cone;#eta;E (GeV)",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book2DHistoVector(iBooker,
                    p_ecalSumVsEta_,
                    "Profile",
                    "ecalSumVsEta",
                    "Avg Ecal Sum in the Iso Cone vs #eta;#eta;E (GeV)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);

  //hcal sum
  book2DHistoVector(
      iBooker, h_hcalSum_, "1D", "hcalSum", "Hcal Sum in the Iso Cone;E (GeV)", sumBin_, sumMin_, sumMax_);
  book2DHistoVector(iBooker,
                    h_hcalSumEBarrel_,
                    "1D",
                    "hcalSumEBarrel",
                    "Hcal Sum in the IsoCone for Barrel;E (GeV)",
                    sumBin_,
                    sumMin_,
                    sumMax_);
  book2DHistoVector(iBooker,
                    h_hcalSumEEndcap_,
                    "1D",
                    "hcalSumEEndcap",
                    "Hcal Sum in the IsoCone for Endcap;E (GeV)",
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_hcalSumVsEt_,
                      "2D",
                      "hcalSumVsEt2D",
                      "Hcal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",
                      reducedEtBin_,
                      etMin_,
                      etMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book3DHistoVector(iBooker,
                    p_hcalSumVsEt_,
                    "Profile",
                    "hcalSumVsEt",
                    "Avg Hcal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",
                    etBin_,
                    etMin_,
                    etMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);
  if (standAlone_) {
    book2DHistoVector(iBooker,
                      h_hcalSumVsEta_,
                      "2D",
                      "hcalSumVsEta2D",
                      "Hcal Sum in the Iso Cone;#eta;E (GeV)",
                      reducedEtaBin_,
                      etaMin_,
                      etaMax_,
                      reducedSumBin_,
                      sumMin_,
                      sumMax_);
  }
  book2DHistoVector(iBooker,
                    p_hcalSumVsEta_,
                    "Profile",
                    "hcalSumVsEta",
                    "Avg Hcal Sum in the Iso Cone vs #eta;#eta;E (GeV)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    sumBin_,
                    sumMin_,
                    sumMax_);

  //h over e
  book3DHistoVector(iBooker, h_hOverE_, "1D", "hOverE", "H/E;H/E", hOverEBin_, hOverEMin_, hOverEMax_);
  book2DHistoVector(iBooker,
                    p_hOverEVsEt_,
                    "Profile",
                    "hOverEVsEt",
                    "Avg H/E vs Et;E_{T} (GeV);H/E",
                    etBin_,
                    etMin_,
                    etMax_,
                    hOverEBin_,
                    hOverEMin_,
                    hOverEMax_);
  book2DHistoVector(iBooker,
                    p_hOverEVsEta_,
                    "Profile",
                    "hOverEVsEta",
                    "Avg H/E vs #eta;#eta;H/E",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    hOverEBin_,
                    hOverEMin_,
                    hOverEMax_);
  book3DHistoVector(iBooker, h_h1OverE_, "1D", "h1OverE", "H/E for Depth 1;H/E", hOverEBin_, hOverEMin_, hOverEMax_);
  book3DHistoVector(iBooker, h_h2OverE_, "1D", "h2OverE", "H/E for Depth 2;H/E", hOverEBin_, hOverEMin_, hOverEMax_);

  // pf isolation
  book2DHistoVector(
      iBooker, h_phoIsoBarrel_, "1D", "phoIsoBarrel", "PF photon iso Barrel;E (GeV)", reducedEtBin_, etMin_, 25.);
  book2DHistoVector(
      iBooker, h_phoIsoEndcap_, "1D", "phoIsoEndcap", "PF photon iso Endcap;E (GeV)", reducedEtBin_, etMin_, 25.);
  book2DHistoVector(iBooker,
                    h_chHadIsoBarrel_,
                    "1D",
                    "chHadIsoBarrel",
                    "PF charged Had iso Barrel;E (GeV)",
                    reducedEtBin_,
                    etMin_,
                    25.);
  book2DHistoVector(iBooker,
                    h_chHadIsoEndcap_,
                    "1D",
                    "chHadIsoEndcap",
                    "PF charged Had iso Endcap;E (GeV)",
                    reducedEtBin_,
                    etMin_,
                    25.);
  book2DHistoVector(iBooker,
                    h_nHadIsoBarrel_,
                    "1D",
                    "neutralHadIsoBarrel",
                    "PF neutral Had iso Barrel;E (GeV)",
                    reducedEtBin_,
                    etMin_,
                    25.);
  book2DHistoVector(iBooker,
                    h_nHadIsoEndcap_,
                    "1D",
                    "neutralHadIsoEndcap",
                    "PF neutral Had iso Endcap;E (GeV)",
                    reducedEtBin_,
                    etMin_,
                    25.);

  //OTHER VARIABLES
  //bad channel histograms
  book2DHistoVector(iBooker,
                    h_phoEt_BadChannels_,
                    "1D",
                    "phoEtBadChannels",
                    "Fraction Containing Bad Channels: E_{T};E_{T} (GeV)",
                    etBin_,
                    etMin_,
                    etMax_);
  book2DHistoVector(iBooker,
                    h_phoEta_BadChannels_,
                    "1D",
                    "phoEtaBadChannels",
                    "Fraction Containing Bad Channels: #eta;#eta",
                    etaBin_,
                    etaMin_,
                    etaMax_);
  book2DHistoVector(iBooker,
                    h_phoPhi_BadChannels_,
                    "1D",
                    "phoPhiBadChannels",
                    "Fraction Containing Bad Channels: #phi;#phi",
                    phiBin_,
                    phiMin_,
                    phiMax_);
}

void PhotonAnalyzer::bookHistogramsConversions(DQMStore::IBooker& iBooker) {
  // Set folder
  iBooker.setCurrentFolder("Egamma/" + fName_ + "/AllPhotons/Et Above 0 GeV/Conversions");

  //ENERGY VARIABLES
  book3DHistoVector(iBooker, h_phoConvE_, "1D", "phoConvE", "E;E (GeV)", eBin_, eMin_, eMax_);
  book3DHistoVector(iBooker, h_phoConvEt_, "1D", "phoConvEt", "E_{T};E_{T} (GeV)", etBin_, etMin_, etMax_);

  //GEOMETRICAL VARIABLES
  book2DHistoVector(iBooker, h_phoConvEta_, "1D", "phoConvEta", "#eta;#eta", etaBin_, etaMin_, etaMax_);
  book3DHistoVector(iBooker, h_phoConvPhi_, "1D", "phoConvPhi", "#phi;#phi", phiBin_, phiMin_, phiMax_);

  //NUMBER OF PHOTONS
  book3DHistoVector(iBooker,
                    h_nConv_,
                    "1D",
                    "nConv",
                    "Number Of Conversions per Event ;# conversions",
                    numberBin_,
                    numberMin_,
                    numberMax_);

  //SHOWER SHAPE VARIABLES
  book3DHistoVector(iBooker, h_phoConvR9_, "1D", "phoConvR9", "R9;R9", r9Bin_, r9Min_, r9Max_);

  //TRACK RELATED VARIABLES
  book3DHistoVector(iBooker, h_eOverPTracks_, "1D", "eOverPTracks", "E/P;E/P", eOverPBin_, eOverPMin_, eOverPMax_);
  book3DHistoVector(iBooker, h_pOverETracks_, "1D", "pOverETracks", "P/E;P/E", eOverPBin_, eOverPMin_, eOverPMax_);
  book3DHistoVector(iBooker,
                    h_dPhiTracksAtVtx_,
                    "1D",
                    "dPhiTracksAtVtx",
                    "#Delta#phi of Tracks at Vertex;#Delta#phi",
                    dPhiTracksBin_,
                    dPhiTracksMin_,
                    dPhiTracksMax_);
  book3DHistoVector(iBooker,
                    h_dPhiTracksAtEcal_,
                    "1D",
                    "dPhiTracksAtEcal",
                    "Abs(#Delta#phi) of Tracks at Ecal;#Delta#phi",
                    dPhiTracksBin_,
                    0.,
                    dPhiTracksMax_);
  book3DHistoVector(iBooker,
                    h_dEtaTracksAtEcal_,
                    "1D",
                    "dEtaTracksAtEcal",
                    "#Delta#eta of Tracks at Ecal;#Delta#eta",
                    dEtaTracksBin_,
                    dEtaTracksMin_,
                    dEtaTracksMax_);
  book3DHistoVector(iBooker,
                    h_dCotTracks_,
                    "1D",
                    "dCotTracks",
                    "#Deltacot(#theta) of Tracks;#Deltacot(#theta)",
                    dEtaTracksBin_,
                    dEtaTracksMin_,
                    dEtaTracksMax_);
  book2DHistoVector(iBooker,
                    p_dCotTracksVsEta_,
                    "Profile",
                    "dCotTracksVsEta",
                    "Avg #Deltacot(#theta) of Tracks vs #eta;#eta;#Deltacot(#theta)",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    dEtaTracksBin_,
                    dEtaTracksMin_,
                    dEtaTracksMax_);
  book2DHistoVector(iBooker,
                    p_nHitsVsEta_,
                    "Profile",
                    "nHitsVsEta",
                    "Avg Number of Hits per Track vs #eta;#eta;# hits",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    etaBin_,
                    0,
                    16);
  book2DHistoVector(
      iBooker, h_tkChi2_, "1D", "tkChi2", "#chi^{2} of Track Fitting;#chi^{2}", chi2Bin_, chi2Min_, chi2Max_);
  book2DHistoVector(iBooker,
                    p_tkChi2VsEta_,
                    "Profile",
                    "tkChi2VsEta",
                    "Avg #chi^{2} of Track Fitting vs #eta;#eta;#chi^{2}",
                    etaBin_,
                    etaMin_,
                    etaMax_,
                    chi2Bin_,
                    chi2Min_,
                    chi2Max_);

  //VERTEX RELATED VARIABLES
  book2DHistoVector(iBooker,
                    h_convVtxRvsZ_,
                    "2D",
                    "convVtxRvsZ",
                    "Vertex Position;Z (cm);R (cm)",
                    500,
                    zMin_,
                    zMax_,
                    rBin_,
                    rMin_,
                    rMax_);
  book2DHistoVector(
      iBooker, h_convVtxZEndcap_, "1D", "convVtxZEndcap", "Vertex Position: #eta > 1.5;Z (cm)", zBin_, zMin_, zMax_);
  book2DHistoVector(iBooker, h_convVtxZ_, "1D", "convVtxZ", "Vertex Position;Z (cm)", zBin_, zMin_, zMax_);
  book2DHistoVector(iBooker, h_convVtxR_, "1D", "convVtxR", "Vertex Position: #eta < 1;R (cm)", rBin_, rMin_, rMax_);
  book2DHistoVector(iBooker,
                    h_convVtxYvsX_,
                    "2D",
                    "convVtxYvsX",
                    "Vertex Position: #eta < 1;X (cm);Y (cm)",
                    xBin_,
                    xMin_,
                    xMax_,
                    yBin_,
                    yMin_,
                    yMax_);
  book2DHistoVector(iBooker,
                    h_vertexChi2Prob_,
                    "1D",
                    "vertexChi2Prob",
                    "#chi^{2} Probability of Vertex Fitting;#chi^{2}",
                    100,
                    0.,
                    1.0);
}

// Booking helper methods:

PhotonAnalyzer::MonitorElement* PhotonAnalyzer::bookHisto(
    DQMStore::IBooker& iBooker, string histoName, string title, int bin, double min, double max) {
  int histo_index = 0;
  stringstream histo_number_stream;

  //determining which folder we're in
  if (iBooker.pwd().find("InvMass") != string::npos) {
    histo_index_invMass_++;
    histo_index = histo_index_invMass_;
  }
  if (iBooker.pwd().find("Efficiencies") != string::npos) {
    histo_index_efficiency_++;
    histo_index = histo_index_efficiency_;
  }

  histo_number_stream << "h_";
  if (histo_index < 10)
    histo_number_stream << "0";
  histo_number_stream << histo_index;

  return iBooker.book1D(histo_number_stream.str() + "_" + histoName, title, bin, min, max);
}

void PhotonAnalyzer::book2DHistoVector(DQMStore::IBooker& iBooker,
                                       vector<vector<MonitorElement*> >& temp2DVector,
                                       string histoType,
                                       string histoName,
                                       string title,
                                       int xbin,
                                       double xmin,
                                       double xmax,
                                       int ybin,
                                       double ymin,
                                       double ymax) {
  int histo_index = 0;

  vector<MonitorElement*> temp1DVector;

  //determining which folder we're in
  bool conversionPlot = false;
  if (iBooker.pwd().find("Conversions") != string::npos)
    conversionPlot = true;
  bool TwoDPlot = false;
  if (histoName.find("2D") != string::npos)
    TwoDPlot = true;

  if (conversionPlot) {
    histo_index_conversions_++;
    histo_index = histo_index_conversions_;
  } else {
    histo_index_photons_++;
    histo_index = histo_index_photons_;
  }

  stringstream histo_number_stream;
  histo_number_stream << "h_";
  if (histo_index < 10)
    histo_number_stream << "0";
  histo_number_stream << histo_index << "_";

  for (int cut = 0; cut != numberOfSteps_; ++cut) {       //looping over Et cut values
    for (uint type = 0; type != types_.size(); ++type) {  //looping over isolation type
      currentFolder_.str("");
      currentFolder_ << "Egamma/" + fName_ + "/" << types_[type] << "Photons/Et above " << (cut + 1) * cutStep_
                     << " GeV";
      if (conversionPlot)
        currentFolder_ << "/Conversions";

      iBooker.setCurrentFolder(currentFolder_.str());

      string kind;
      if (conversionPlot)
        kind = " Conversions: ";
      else
        kind = " Photons: ";

      if (histoType == "1D")
        temp1DVector.push_back(
            iBooker.book1D(histo_number_stream.str() + histoName, types_[type] + kind + title, xbin, xmin, xmax));
      else if (histoType == "2D") {
        if ((TwoDPlot && type == 0) || !TwoDPlot) {  //only book the 2D plots in the "AllPhotons" folder
          temp1DVector.push_back(iBooker.book2D(
              histo_number_stream.str() + histoName, types_[type] + kind + title, xbin, xmin, xmax, ybin, ymin, ymax));
        }
      } else if (histoType == "Profile")
        temp1DVector.push_back(iBooker.bookProfile(
            histo_number_stream.str() + histoName, types_[type] + kind + title, xbin, xmin, xmax, ybin, ymin, ymax, ""));
      else
        cout << "bad histoType\n";
    }

    temp2DVector.push_back(temp1DVector);
    temp1DVector.clear();
  }
}

void PhotonAnalyzer::book3DHistoVector(DQMStore::IBooker& iBooker,
                                       vector<vector<vector<MonitorElement*> > >& temp3DVector,
                                       string histoType,
                                       string histoName,
                                       string title,
                                       int xbin,
                                       double xmin,
                                       double xmax,
                                       int ybin,
                                       double ymin,
                                       double ymax) {
  int histo_index = 0;

  vector<MonitorElement*> temp1DVector;
  vector<vector<MonitorElement*> > temp2DVector;

  //determining which folder we're in
  bool conversionPlot = false;
  if (iBooker.pwd().find("Conversions") != string::npos)
    conversionPlot = true;

  if (conversionPlot) {
    histo_index_conversions_++;
    histo_index = histo_index_conversions_;
  } else {
    histo_index_photons_++;
    histo_index = histo_index_photons_;
  }

  stringstream histo_number_stream;
  histo_number_stream << "h_";
  if (histo_index < 10)
    histo_number_stream << "0";
  histo_number_stream << histo_index << "_";

  for (int cut = 0; cut != numberOfSteps_; ++cut) {         //looping over Et cut values
    for (uint type = 0; type != types_.size(); ++type) {    //looping over isolation type
      for (uint part = 0; part != parts_.size(); ++part) {  //looping over different parts of the ecal
        currentFolder_.str("");
        currentFolder_ << "Egamma/" + fName_ + "/" << types_[type] << "Photons/Et above " << (cut + 1) * cutStep_
                       << " GeV";
        if (conversionPlot)
          currentFolder_ << "/Conversions";
        iBooker.setCurrentFolder(currentFolder_.str());

        string kind;
        if (conversionPlot)
          kind = " Conversions: ";
        else
          kind = " Photons: ";

        if (histoType == "1D")
          temp1DVector.push_back(iBooker.book1D(histo_number_stream.str() + histoName + parts_[part],
                                                types_[type] + kind + parts_[part] + ": " + title,
                                                xbin,
                                                xmin,
                                                xmax));
        else if (histoType == "2D")
          temp1DVector.push_back(iBooker.book2D(histo_number_stream.str() + histoName + parts_[part],
                                                types_[type] + kind + parts_[part] + ": " + title,
                                                xbin,
                                                xmin,
                                                xmax,
                                                ybin,
                                                ymin,
                                                ymax));
        else if (histoType == "Profile")
          temp1DVector.push_back(iBooker.bookProfile(histo_number_stream.str() + histoName + parts_[part],
                                                     types_[type] + kind + parts_[part] + ": " + title,
                                                     xbin,
                                                     xmin,
                                                     xmax,
                                                     ybin,
                                                     ymin,
                                                     ymax,
                                                     ""));
        else
          cout << "bad histoType\n";
      }
      temp2DVector.push_back(temp1DVector);
      temp1DVector.clear();
    }
    temp3DVector.push_back(temp2DVector);
    temp2DVector.clear();
  }
}

// Analysis:

void PhotonAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& esup) {
  using namespace edm;

  if (nEvt_ % prescaleFactor_)
    return;
  nEvt_++;
  LogInfo(fName_) << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ << "\n";

  // Get the trigger results
  bool validTriggerEvent = true;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  const trigger::TriggerEvent dummyTE;
  e.getByToken(triggerEvent_token_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: triggerEvent_" << endl;
    validTriggerEvent = false;
  }
  const trigger::TriggerEvent& triggerEvent(validTriggerEvent ? *(triggerEventHandle.product()) : dummyTE);

  // Get the reconstructed photons
  //  bool validPhotons=true;
  Handle<reco::PhotonCollection> photonHandle;
  e.getByToken(photon_token_, photonHandle);
  if (!photonHandle.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: photon_token_" << endl;
    // validPhotons=false;
  }
  const reco::PhotonCollection& photonCollection(*(photonHandle.product()));

  // Get the PhotonId objects
  //  bool validloosePhotonID=true;
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  e.getByToken(PhotonIDLoose_token_, loosePhotonFlag);
  if (!loosePhotonFlag.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: PhotonIDLoose_token_" << endl;
    //    validloosePhotonID=false;
  }
  //  edm::ValueMap<bool> dummyLPID;
  //  const edm::ValueMap<bool>& loosePhotonID(validloosePhotonID? *(loosePhotonFlag.product()) : dummyLPID);

  //  bool validtightPhotonID=true;
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  e.getByToken(PhotonIDTight_token_, tightPhotonFlag);
  if (!tightPhotonFlag.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: PhotonIDTight_token_" << endl;
    //    validtightPhotonID=false;
  }
  //  edm::ValueMap<bool> dummyTPI;
  //  const edm::ValueMap<bool>& tightPhotonID(validtightPhotonID ? *(tightPhotonFlag.product()) : dummyTPI);

  edm::Handle<reco::VertexCollection> vtxH;
  if (!isHeavyIon_) {
    e.getByToken(offline_pvToken_, vtxH);
    h_nRecoVtx_->Fill(float(vtxH->size()));
  }

  // Create array to hold #photons/event information
  int nPho[100][3][3];

  for (int cut = 0; cut != 100; ++cut) {
    for (unsigned int type = 0; type != types_.size(); ++type) {
      for (unsigned int part = 0; part != parts_.size(); ++part) {
        nPho[cut][type][part] = 0;
      }
    }
  }
  // Create array to hold #conversions/event information
  int nConv[100][3][3];

  for (int cut = 0; cut != 100; ++cut) {
    for (unsigned int type = 0; type != types_.size(); ++type) {
      for (unsigned int part = 0; part != parts_.size(); ++part) {
        nConv[cut][type][part] = 0;
      }
    }
  }

  //Prepare list of photon-related HLT filter names
  vector<int> Keys;

  for (uint filterIndex = 0; filterIndex < triggerEvent.sizeFilters();
       ++filterIndex) {  //loop over all trigger filters in event (i.e. filters passed)
    string label = triggerEvent.filterTag(filterIndex).label();
    if (label.find("Photon") != string::npos) {  //get photon-related filters
      for (uint filterKeyIndex = 0; filterKeyIndex < triggerEvent.filterKeys(filterIndex).size();
           ++filterKeyIndex) {  //loop over keys to objects passing this filter
        Keys.push_back(
            triggerEvent.filterKeys(filterIndex)[filterKeyIndex]);  //add keys to a vector for later reference
      }
    }
  }

  // sort Keys vector in ascending order
  // and erases duplicate entries from the vector
  sort(Keys.begin(), Keys.end());
  for (uint i = 0; i < Keys.size();) {
    if (i != (Keys.size() - 1)) {
      if (Keys[i] == Keys[i + 1])
        Keys.erase(Keys.begin() + i + 1);
      else
        ++i;
    } else
      ++i;
  }

  //We now have a vector of unique keys to TriggerObjects passing a photon-related filter

  // old int photonCounter = 0;

  /////////////////////////BEGIN LOOP OVER THE COLLECTION OF PHOTONS IN THE EVENT/////////////////////////
  for (unsigned int iPho = 0; iPho < photonHandle->size(); iPho++) {
    const reco::Photon* aPho = &photonCollection[iPho];
    //  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {

    //for HLT efficiency plots

    h_phoEta_preHLT_->Fill(aPho->eta());
    h_phoEt_preHLT_->Fill(aPho->et());

    double deltaR = 1000.;
    double deltaRMin = 1000.;
    double deltaRMax = 0.05;  //sets deltaR threshold for matching photons to trigger objects

    for (vector<int>::const_iterator objectKey = Keys.begin(); objectKey != Keys.end();
         objectKey++) {  //loop over keys to objects that fired photon triggers

      deltaR = reco::deltaR(triggerEvent.getObjects()[(*objectKey)].eta(),
                            triggerEvent.getObjects()[(*objectKey)].phi(),
                            aPho->superCluster()->eta(),
                            aPho->superCluster()->phi());
      if (deltaR < deltaRMin)
        deltaRMin = deltaR;
    }

    if (deltaRMin > deltaRMax) {  //photon fails delta R cut
      if (useTriggerFiltering_)
        continue;  //throw away photons that haven't passed any photon filters
    }

    if (deltaRMin <= deltaRMax) {  //photon passes delta R cut
      h_phoEta_postHLT_->Fill(aPho->eta());
      h_phoEt_postHLT_->Fill(aPho->et());
    }

    //    if (aPho->et()  < minPhoEtCut_) continue;
    bool isLoosePhoton(false), isTightPhoton(false);
    if (photonSelection(aPho))
      isLoosePhoton = true;

    //find out which part of the Ecal contains the photon
    bool phoIsInBarrel = false;
    bool phoIsInEndcap = false;
    float etaPho = aPho->superCluster()->eta();
    if (fabs(etaPho) < 1.479)
      phoIsInBarrel = true;
    else {
      phoIsInEndcap = true;
    }

    int part = 0;
    if (phoIsInBarrel)
      part = 1;
    if (phoIsInEndcap)
      part = 2;

    /////  From 30X on, Photons are already pre-selected at reconstruction level with a looseEM isolation
    bool isIsolated = false;
    if (isolationStrength_ == 0)
      isIsolated = isLoosePhoton;
    if (isolationStrength_ == 1)
      isIsolated = isTightPhoton;
    if (isolationStrength_ == 2)
      isIsolated = photonSelectionSlimmed(aPho);

    int type = 0;
    if (isIsolated)
      type = 1;
    if (!excludeBkgHistos_ && !isIsolated)
      type = 2;

    //get rechit collection containing this photon
    bool validEcalRecHits = true;
    edm::Handle<EcalRecHitCollection> ecalRecHitHandle;
    EcalRecHitCollection ecalRecHitCollection;
    if (phoIsInBarrel) {
      // Get handle to barrel rec hits
      e.getByToken(barrelRecHit_token_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
        edm::LogError(fName_) << "Error! Can't get the product: barrelRecHit_token_";
        validEcalRecHits = false;
      }
    } else if (phoIsInEndcap) {
      // Get handle to endcap rec hits
      e.getByToken(endcapRecHit_token_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
        edm::LogError(fName_) << "Error! Can't get the product: endcapRecHit_token";
        validEcalRecHits = false;
      }
    }
    if (validEcalRecHits)
      ecalRecHitCollection = *(ecalRecHitHandle.product());

    //if (aPho->isEBEEGap()) continue;  //cut out gap photons

    //filling histograms to make isolation efficiencies
    if (isLoosePhoton) {
      h_phoEta_Loose_->Fill(aPho->eta());
      h_phoEt_Loose_->Fill(aPho->et());
    }
    if (isTightPhoton) {
      h_phoEta_Tight_->Fill(aPho->eta());
      h_phoEt_Tight_->Fill(aPho->et());
    }

    for (int cut = 0; cut != numberOfSteps_; ++cut) {  //loop over different transverse energy cuts
      double Et = aPho->et();
      bool passesCuts = false;

      //sorting the photon into the right Et-dependant folder
      if (useBinning_ && Et > (cut + 1) * cutStep_ && ((Et < (cut + 2) * cutStep_) | (cut == numberOfSteps_ - 1))) {
        passesCuts = true;
      } else if (!useBinning_ && Et > (cut + 1) * cutStep_) {
        passesCuts = true;
      }

      if (passesCuts) {
        //filling isolation variable histograms

        //tracker isolation variables
        fill2DHistoVector(h_nTrackIsolSolid_, aPho->nTrkSolidConeDR04(), cut, type);
        fill2DHistoVector(h_nTrackIsolHollow_, aPho->nTrkHollowConeDR04(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_nTrackIsolSolidVsEta_, aPho->eta(), aPho->nTrkSolidConeDR04(), cut, type);
        fill2DHistoVector(p_nTrackIsolSolidVsEta_, aPho->eta(), aPho->nTrkSolidConeDR04(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_nTrackIsolHollowVsEta_, aPho->eta(), aPho->nTrkHollowConeDR04(), cut, type);
        fill2DHistoVector(p_nTrackIsolHollowVsEta_, aPho->eta(), aPho->nTrkHollowConeDR04(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_nTrackIsolSolidVsEt_, aPho->et(), aPho->nTrkSolidConeDR04(), cut, type);
        fill2DHistoVector(p_nTrackIsolSolidVsEt_, aPho->et(), aPho->nTrkSolidConeDR04(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_nTrackIsolHollowVsEt_, aPho->et(), aPho->nTrkHollowConeDR04(), cut, type);
        fill2DHistoVector(p_nTrackIsolHollowVsEt_, aPho->et(), aPho->nTrkHollowConeDR04(), cut, type);

        ///////
        fill2DHistoVector(h_trackPtSumSolid_, aPho->trkSumPtSolidConeDR04(), cut, type);
        fill2DHistoVector(h_trackPtSumHollow_, aPho->trkSumPtSolidConeDR04(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_trackPtSumSolidVsEta_, aPho->eta(), aPho->trkSumPtSolidConeDR04(), cut, type);
        fill2DHistoVector(p_trackPtSumSolidVsEta_, aPho->eta(), aPho->trkSumPtSolidConeDR04(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_trackPtSumHollowVsEta_, aPho->eta(), aPho->trkSumPtHollowConeDR04(), cut, type);
        fill2DHistoVector(p_trackPtSumHollowVsEta_, aPho->eta(), aPho->trkSumPtHollowConeDR04(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_trackPtSumSolidVsEt_, aPho->et(), aPho->trkSumPtSolidConeDR04(), cut, type);
        fill2DHistoVector(p_trackPtSumSolidVsEt_, aPho->et(), aPho->trkSumPtSolidConeDR04(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_trackPtSumHollowVsEt_, aPho->et(), aPho->trkSumPtHollowConeDR04(), cut, type);
        fill2DHistoVector(p_trackPtSumHollowVsEt_, aPho->et(), aPho->trkSumPtHollowConeDR04(), cut, type);
        //calorimeter isolation variables

        fill2DHistoVector(h_ecalSum_, aPho->ecalRecHitSumEtConeDR04(), cut, type);
        if (aPho->isEB()) {
          fill2DHistoVector(h_ecalSumEBarrel_, aPho->ecalRecHitSumEtConeDR04(), cut, type);
        }
        if (aPho->isEE()) {
          fill2DHistoVector(h_ecalSumEEndcap_, aPho->ecalRecHitSumEtConeDR04(), cut, type);
        }
        if (standAlone_)
          fill2DHistoVector(h_ecalSumVsEta_, aPho->eta(), aPho->ecalRecHitSumEtConeDR04(), cut, type);
        fill2DHistoVector(p_ecalSumVsEta_, aPho->eta(), aPho->ecalRecHitSumEtConeDR04(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_ecalSumVsEt_, aPho->et(), aPho->ecalRecHitSumEtConeDR04(), cut, type);
        fill3DHistoVector(p_ecalSumVsEt_, aPho->et(), aPho->ecalRecHitSumEtConeDR04(), cut, type, part);

        ///////

        fill2DHistoVector(h_hcalSum_, aPho->hcalTowerSumEtConeDR04(), cut, type);
        if (aPho->isEB()) {
          fill2DHistoVector(h_hcalSumEBarrel_, aPho->hcalTowerSumEtConeDR04(), cut, type);
        }
        if (aPho->isEE()) {
          fill2DHistoVector(h_hcalSumEEndcap_, aPho->hcalTowerSumEtConeDR04(), cut, type);
        }
        if (standAlone_)
          fill2DHistoVector(h_hcalSumVsEta_, aPho->eta(), aPho->hcalTowerSumEtConeDR04(), cut, type);
        fill2DHistoVector(p_hcalSumVsEta_, aPho->eta(), aPho->hcalTowerSumEtConeDR04(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_hcalSumVsEt_, aPho->et(), aPho->hcalTowerSumEtConeDR04(), cut, type);
        fill3DHistoVector(p_hcalSumVsEt_, aPho->et(), aPho->hcalTowerSumEtConeDR04(), cut, type, part);

        fill3DHistoVector(h_hOverE_, aPho->hadronicOverEm(), cut, type, part);
        fill2DHistoVector(p_hOverEVsEta_, aPho->eta(), aPho->hadronicOverEm(), cut, type);
        fill2DHistoVector(p_hOverEVsEt_, aPho->et(), aPho->hadronicOverEm(), cut, type);

        fill3DHistoVector(h_h1OverE_, aPho->hadronicOverEm(1), cut, type, part);
        fill3DHistoVector(h_h2OverE_, aPho->hadronicOverEm(2), cut, type, part);

        // filling pf isolation variables
        if (aPho->isEB()) {
          fill2DHistoVector(h_phoIsoBarrel_, aPho->photonIso(), cut, type);
          fill2DHistoVector(h_chHadIsoBarrel_, aPho->chargedHadronIso(), cut, type);
          fill2DHistoVector(h_nHadIsoBarrel_, aPho->neutralHadronIso(), cut, type);
        }
        if (aPho->isEE()) {
          fill2DHistoVector(h_phoIsoEndcap_, aPho->photonIso(), cut, type);
          fill2DHistoVector(h_chHadIsoEndcap_, aPho->chargedHadronIso(), cut, type);
          fill2DHistoVector(h_nHadIsoEndcap_, aPho->neutralHadronIso(), cut, type);
        }

        //filling photon histograms
        nPho[cut][0][0]++;
        nPho[cut][0][part]++;
        if (type != 0) {
          nPho[cut][type][0]++;
          nPho[cut][type][part]++;
        }

        //energy variables

        fill3DHistoVector(h_phoE_, aPho->energy(), cut, type, part);
        fill3DHistoVector(h_phoSigmaEoverE_,
                          aPho->getCorrectedEnergyError(aPho->getCandidateP4type()) / aPho->energy(),
                          cut,
                          type,
                          part);

        if (!isHeavyIon_)
          fill3DHistoVector(p_phoSigmaEoverEvsNVtx_,
                            float(vtxH->size()),
                            aPho->getCorrectedEnergyError(aPho->getCandidateP4type()) / aPho->energy(),
                            cut,
                            type,
                            part);

        fill3DHistoVector(h_phoEt_, aPho->et(), cut, type, part);

        //geometrical variables

        fill2DHistoVector(h_phoEta_, aPho->eta(), cut, type);
        fill2DHistoVector(h_scEta_, aPho->superCluster()->eta(), cut, type);

        fill3DHistoVector(h_phoPhi_, aPho->phi(), cut, type, part);
        fill3DHistoVector(h_scPhi_, aPho->superCluster()->phi(), cut, type, part);

        //shower shape variables

        fill3DHistoVector(h_r9_, aPho->r9(), cut, type, part);
        if (standAlone_)
          fill2DHistoVector(h_r9VsEta_, aPho->eta(), aPho->r9(), cut, type);
        fill2DHistoVector(p_r9VsEta_, aPho->eta(), aPho->r9(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_r9VsEt_, aPho->et(), aPho->r9(), cut, type);
        fill2DHistoVector(p_r9VsEt_, aPho->et(), aPho->r9(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_e1x5VsEta_, aPho->eta(), aPho->e1x5(), cut, type);
        fill2DHistoVector(p_e1x5VsEta_, aPho->eta(), aPho->e1x5(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_e1x5VsEt_, aPho->et(), aPho->e1x5(), cut, type);
        fill2DHistoVector(p_e1x5VsEt_, aPho->et(), aPho->e1x5(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_e2x5VsEta_, aPho->eta(), aPho->e2x5(), cut, type);
        fill2DHistoVector(p_e2x5VsEta_, aPho->eta(), aPho->e2x5(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_e2x5VsEt_, aPho->et(), aPho->e2x5(), cut, type);
        fill2DHistoVector(p_e2x5VsEt_, aPho->et(), aPho->e2x5(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_maxEXtalOver3x3VsEta_, aPho->eta(), aPho->maxEnergyXtal() / aPho->e3x3(), cut, type);
        fill2DHistoVector(p_maxEXtalOver3x3VsEta_, aPho->eta(), aPho->maxEnergyXtal() / aPho->e3x3(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_maxEXtalOver3x3VsEt_, aPho->et(), aPho->maxEnergyXtal() / aPho->e3x3(), cut, type);
        fill2DHistoVector(p_maxEXtalOver3x3VsEt_, aPho->et(), aPho->maxEnergyXtal() / aPho->e3x3(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_r1x5VsEta_, aPho->eta(), aPho->r1x5(), cut, type);
        fill2DHistoVector(p_r1x5VsEta_, aPho->eta(), aPho->r1x5(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_r1x5VsEt_, aPho->et(), aPho->r1x5(), cut, type);
        fill2DHistoVector(p_r1x5VsEt_, aPho->et(), aPho->r1x5(), cut, type);

        if (standAlone_)
          fill2DHistoVector(h_r2x5VsEta_, aPho->eta(), aPho->r2x5(), cut, type);
        fill2DHistoVector(p_r2x5VsEta_, aPho->eta(), aPho->r2x5(), cut, type);
        if (standAlone_)
          fill2DHistoVector(h_r2x5VsEt_, aPho->et(), aPho->r2x5(), cut, type);
        fill2DHistoVector(p_r2x5VsEt_, aPho->et(), aPho->r2x5(), cut, type);

        fill3DHistoVector(h_phoSigmaIetaIeta_, aPho->sigmaIetaIeta(), cut, type, part);
        if (standAlone_)
          fill2DHistoVector(h_sigmaIetaIetaVsEta_, aPho->eta(), aPho->sigmaIetaIeta(), cut, type);
        fill2DHistoVector(p_sigmaIetaIetaVsEta_, aPho->eta(), aPho->sigmaIetaIeta(), cut, type);

        //filling histograms for photons containing a bad ECAL channel
        bool atLeastOneDeadChannel = false;
        for (reco::CaloCluster_iterator bcIt = aPho->superCluster()->clustersBegin();
             bcIt != aPho->superCluster()->clustersEnd();
             ++bcIt) {  //loop over basic clusters in SC
          for (vector<pair<DetId, float> >::const_iterator rhIt = (*bcIt)->hitsAndFractions().begin();
               rhIt != (*bcIt)->hitsAndFractions().end();
               ++rhIt) {  //loop over rec hits in basic cluster

            for (EcalRecHitCollection::const_iterator it = ecalRecHitCollection.begin();
                 it != ecalRecHitCollection.end();
                 ++it) {                        //loop over all rec hits to find the right ones
              if (rhIt->first == (*it).id()) {  //found the matching rechit
                if ((*it).recoFlag() == 9) {    //has a bad channel
                  atLeastOneDeadChannel = true;
                  break;
                }
              }
            }
          }
        }
        if (atLeastOneDeadChannel) {
          fill2DHistoVector(h_phoPhi_BadChannels_, aPho->phi(), cut, type);
          fill2DHistoVector(h_phoEta_BadChannels_, aPho->eta(), cut, type);
          fill2DHistoVector(h_phoEt_BadChannels_, aPho->et(), cut, type);
        }

        // filling conversion-related histograms
        if (aPho->hasConversionTracks()) {
          nConv[cut][0][0]++;
          nConv[cut][0][part]++;
          nConv[cut][type][0]++;
          nConv[cut][type][part]++;
        }

        //loop over conversions (don't forget, we're still inside the photon loop,
        // i.e. these are all the conversions for this ONE photon, not for all the photons in the event)
        reco::ConversionRefVector conversions = aPho->conversions();
        for (unsigned int iConv = 0; iConv < conversions.size(); iConv++) {
          reco::ConversionRef aConv = conversions[iConv];

          if (aConv->nTracks() < 2)
            continue;

          //fill histogram for denominator of vertex reconstruction efficiency plot
          if (cut == 0)
            h_phoEta_Vertex_->Fill(aConv->refittedPairMomentum().eta());

          if (!(aConv->conversionVertex().isValid()))
            continue;

          float chi2Prob = ChiSquaredProbability(aConv->conversionVertex().chi2(), aConv->conversionVertex().ndof());

          if (chi2Prob < 0.0005)
            continue;

          fill2DHistoVector(h_vertexChi2Prob_, chi2Prob, cut, type);

          fill3DHistoVector(h_phoConvE_, aPho->energy(), cut, type, part);
          fill3DHistoVector(h_phoConvEt_, aPho->et(), cut, type, part);
          fill3DHistoVector(h_phoConvR9_, aPho->r9(), cut, type, part);

          if (cut == 0 && isLoosePhoton) {
            h_convEta_Loose_->Fill(aPho->eta());
            h_convEt_Loose_->Fill(aPho->et());
          }
          if (cut == 0 && isTightPhoton) {
            h_convEta_Tight_->Fill(aPho->eta());
            h_convEt_Tight_->Fill(aPho->et());
          }

          fill2DHistoVector(h_phoConvEta_, aConv->refittedPairMomentum().eta(), cut, type);
          fill3DHistoVector(h_phoConvPhi_, aConv->refittedPairMomentum().phi(), cut, type, part);

          //we use the photon position because we'll be dividing it by a photon histogram (not a conversion histogram)
          fill2DHistoVector(h_phoConvEtaForEfficiency_, aPho->eta(), cut, type);
          fill3DHistoVector(h_phoConvPhiForEfficiency_, aPho->phi(), cut, type, part);

          //vertex histograms
          double convR = sqrt(aConv->conversionVertex().position().perp2());
          double scalar = aConv->conversionVertex().position().x() * aConv->refittedPairMomentum().x() +
                          aConv->conversionVertex().position().y() * aConv->refittedPairMomentum().y();
          if (scalar < 0)
            convR = -convR;

          fill2DHistoVector(h_convVtxRvsZ_,
                            aConv->conversionVertex().position().z(),
                            convR,
                            cut,
                            type);  //trying to "see" R-Z view of tracker
          fill2DHistoVector(h_convVtxZ_, aConv->conversionVertex().position().z(), cut, type);

          if (fabs(aPho->eta()) > 1.5) {  //trying to "see" tracker endcaps
            fill2DHistoVector(h_convVtxZEndcap_, aConv->conversionVertex().position().z(), cut, type);
          } else if (fabs(aPho->eta()) < 1) {  //trying to "see" tracker barrel
            fill2DHistoVector(h_convVtxR_, convR, cut, type);
            fill2DHistoVector(h_convVtxYvsX_,
                              aConv->conversionVertex().position().x(),
                              aConv->conversionVertex().position().y(),
                              cut,
                              type);
          }

          const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();

          for (unsigned int i = 0; i < tracks.size(); i++) {
            fill2DHistoVector(h_tkChi2_, tracks[i]->normalizedChi2(), cut, type);
            fill2DHistoVector(p_tkChi2VsEta_, aPho->eta(), tracks[i]->normalizedChi2(), cut, type);
            fill2DHistoVector(p_dCotTracksVsEta_, aPho->eta(), aConv->pairCotThetaSeparation(), cut, type);
            fill2DHistoVector(p_nHitsVsEta_, aPho->eta(), float(tracks[i]->numberOfValidHits()), cut, type);
          }

          //calculating delta eta and delta phi of the two tracks

          float DPhiTracksAtVtx = -99;
          float dPhiTracksAtEcal = -99;
          float dEtaTracksAtEcal = -99;

          float phiTk1 = aConv->tracksPin()[0].phi();
          float phiTk2 = aConv->tracksPin()[1].phi();
          DPhiTracksAtVtx = phiTk1 - phiTk2;
          DPhiTracksAtVtx = phiNormalization(DPhiTracksAtVtx);

          if (!aConv->bcMatchingWithTracks().empty() && aConv->bcMatchingWithTracks()[0].isNonnull() &&
              aConv->bcMatchingWithTracks()[1].isNonnull()) {
            float recoPhi1 = aConv->ecalImpactPosition()[0].phi();
            float recoPhi2 = aConv->ecalImpactPosition()[1].phi();
            float recoEta1 = aConv->ecalImpactPosition()[0].eta();
            float recoEta2 = aConv->ecalImpactPosition()[1].eta();

            recoPhi1 = phiNormalization(recoPhi1);
            recoPhi2 = phiNormalization(recoPhi2);

            dPhiTracksAtEcal = recoPhi1 - recoPhi2;
            dPhiTracksAtEcal = phiNormalization(dPhiTracksAtEcal);
            dEtaTracksAtEcal = recoEta1 - recoEta2;
          }

          fill3DHistoVector(h_dPhiTracksAtVtx_, DPhiTracksAtVtx, cut, type, part);
          fill3DHistoVector(h_dPhiTracksAtEcal_, fabs(dPhiTracksAtEcal), cut, type, part);
          fill3DHistoVector(h_dEtaTracksAtEcal_, dEtaTracksAtEcal, cut, type, part);
          fill3DHistoVector(h_eOverPTracks_, aConv->EoverPrefittedTracks(), cut, type, part);
          fill3DHistoVector(h_pOverETracks_, 1. / aConv->EoverPrefittedTracks(), cut, type, part);
          fill3DHistoVector(h_dCotTracks_, aConv->pairCotThetaSeparation(), cut, type, part);
        }  //end loop over conversions
      }    //end loop over photons passing cuts
    }      //end loop over transverse energy cuts

    //make invariant mass plots

    if (isIsolated && aPho->et() >= invMassEtCut_) {
      for (unsigned int iPho2 = iPho + 1; iPho2 < photonHandle->size(); iPho2++) {
        const reco::Photon* aPho2 = &photonCollection[iPho2];

        //      for (reco::PhotonCollection::const_iterator iPho2=iPho+1; iPho2!=photonCollection.end(); iPho2++){

        //	edm::Ref<reco::PhotonCollection> photonref2(photonHandle, photonCounter); //note: it's correct to use photonCounter and not photonCounter+1
        //since it has already been incremented earlier

        bool isTightPhoton2(false), isLoosePhoton2(false);
        if (photonSelection(aPho2))
          isLoosePhoton2 = true;

        // Old if ( !isHeavyIon_ ) {
        //  isTightPhoton2 = (tightPhotonID)[aPho2];
        // isLoosePhoton2 = (loosePhotonID)[aPho2];
        //	}

        bool isIsolated2 = false;
        if (isolationStrength_ == 0)
          isIsolated2 = isLoosePhoton2;
        if (isolationStrength_ == 1)
          isIsolated2 = isTightPhoton2;
        if (isolationStrength_ == 2)
          isIsolated2 = photonSelectionSlimmed(aPho2);

        reco::ConversionRefVector conversions = aPho->conversions();
        reco::ConversionRefVector conversions2 = aPho2->conversions();

        if (isIsolated2 && aPho2->et() >= invMassEtCut_) {
          math::XYZTLorentzVector p12 = aPho->p4() + aPho2->p4();
          float gamgamMass2 = p12.Dot(p12);

          h_invMassAllPhotons_->Fill(sqrt(gamgamMass2));
          if (aPho->isEB() && aPho2->isEB()) {
            h_invMassPhotonsEBarrel_->Fill(sqrt(gamgamMass2));
          } else if (aPho->isEE() && aPho2->isEE()) {
            h_invMassPhotonsEEndcap_->Fill(sqrt(gamgamMass2));
          } else {
            h_invMassPhotonsEEndcapEBarrel_->Fill(sqrt(gamgamMass2));
          }

          if (!conversions.empty() && conversions[0]->nTracks() >= 2) {
            if (!conversions2.empty() && conversions2[0]->nTracks() >= 2)
              h_invMassTwoWithTracks_->Fill(sqrt(gamgamMass2));
            else
              h_invMassOneWithTracks_->Fill(sqrt(gamgamMass2));
          } else if (!conversions2.empty() && conversions2[0]->nTracks() >= 2)
            h_invMassOneWithTracks_->Fill(sqrt(gamgamMass2));
          else
            h_invMassZeroWithTracks_->Fill(sqrt(gamgamMass2));
        }
      }
    }
  }  /// End loop over Reco photons

  //filling number of photons/conversions per event histograms
  for (int cut = 0; cut != numberOfSteps_; ++cut) {
    for (uint type = 0; type != types_.size(); ++type) {
      for (uint part = 0; part != parts_.size(); ++part) {
        h_nPho_[cut][type][part]->Fill(float(nPho[cut][type][part]));
        h_nConv_[cut][type][part]->Fill(float(nConv[cut][type][part]));
      }
    }
  }
}  //End of Analyze method

////////////BEGIN AUXILIARY FUNCTIONS//////////////

float PhotonAnalyzer::phiNormalization(float& phi) {
  const float PI = 3.1415927;
  const float TWOPI = 2.0 * PI;

  if (phi > PI) {
    phi = phi - TWOPI;
  }
  if (phi < -PI) {
    phi = phi + TWOPI;
  }

  return phi;
}

void PhotonAnalyzer::fill2DHistoVector(
    vector<vector<MonitorElement*> >& histoVector, double x, double y, int cut, int type) {
  histoVector[cut][0]->Fill(x, y);
  if (histoVector[cut].size() > 1)
    histoVector[cut][type]->Fill(x, y);  //don't try to fill 2D histos that are only in the "AllPhotons" folder
}

void PhotonAnalyzer::fill2DHistoVector(vector<vector<MonitorElement*> >& histoVector, double x, int cut, int type) {
  histoVector[cut][0]->Fill(x);
  histoVector[cut][type]->Fill(x);
}

void PhotonAnalyzer::fill3DHistoVector(
    vector<vector<vector<MonitorElement*> > >& histoVector, double x, int cut, int type, int part) {
  histoVector[cut][0][0]->Fill(x);
  histoVector[cut][0][part]->Fill(x);
  histoVector[cut][type][0]->Fill(x);
  histoVector[cut][type][part]->Fill(x);
}

void PhotonAnalyzer::fill3DHistoVector(
    vector<vector<vector<MonitorElement*> > >& histoVector, double x, double y, int cut, int type, int part) {
  histoVector[cut][0][0]->Fill(x, y);
  histoVector[cut][0][part]->Fill(x, y);
  histoVector[cut][type][0]->Fill(x, y);
  histoVector[cut][type][part]->Fill(x, y);
}

bool PhotonAnalyzer::photonSelection(const reco::Photon* pho) {
  bool result = true;
  if (pho->pt() < minPhoEtCut_)
    result = false;
  if (fabs(pho->eta()) > photonMaxEta_)
    result = false;
  if (pho->isEBEEGap())
    result = false;

  double EtCorrHcalIso = pho->hcalTowerSumEtConeDR03() - 0.005 * pho->pt();
  double EtCorrTrkIso = pho->trkSumPtHollowConeDR03() - 0.002 * pho->pt();

  if (pho->r9() <= 0.9) {
    if (pho->isEB() && (pho->hadTowOverEm() > 0.075 || pho->sigmaIetaIeta() > 0.014))
      result = false;
    if (pho->isEE() && (pho->hadTowOverEm() > 0.075 || pho->sigmaIetaIeta() > 0.034))
      result = false;
    ///  remove after moriond    if (EtCorrEcalIso>4.0) result=false;
    if (EtCorrHcalIso > 4.0)
      result = false;
    if (EtCorrTrkIso > 4.0)
      result = false;
    if (pho->chargedHadronIso() > 4)
      result = false;
  } else {
    if (pho->isEB() && (pho->hadTowOverEm() > 0.082 || pho->sigmaIetaIeta() > 0.014))
      result = false;
    if (pho->isEE() && (pho->hadTowOverEm() > 0.075 || pho->sigmaIetaIeta() > 0.034))
      result = false;
    /// remove after moriond if (EtCorrEcalIso>50.0) result=false;
    if (EtCorrHcalIso > 50.0)
      result = false;
    if (EtCorrTrkIso > 50.0)
      result = false;
    if (pho->chargedHadronIso() > 4)
      result = false;
  }
  return result;
}

bool PhotonAnalyzer::photonSelectionSlimmed(const reco::Photon* pho) {
  bool result = true;

  if (pho->pt() < minPhoEtCut_)
    result = false;
  if (fabs(pho->eta()) > photonMaxEta_)
    result = false;
  if (pho->isEBEEGap())
    result = false;

  return result;
}

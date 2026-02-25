#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "PhysicsTools/SelectorUtils/interface/Run3ScoutingPFJetIDSelectionFunctor.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TLorentzVector.h"

class JetMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  JetMonitor(const edm::ParameterSet&);
  ~JetMonitor() throw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  struct correctedPFJets {
    double pt;
    double eta;
    double phi;
    double NHF;
    double NEMF;
    double CHF;
    double CEMF;
    double MUF;
    int NumNeutralParticles;
    int CHM;
  };
  std::vector<correctedPFJets> corrected_jets;

  bool passTightJetID(const correctedPFJets& jet);
  bool isCleanJet(double JetEta, double JetPhi, const std::vector<reco::Muon>& muons, double dr2Cut);
  bool isGoodScoutingMuon(Run3ScoutingMuon const& scoutingMuon);
  bool isCleanScoutingJet(double ScoutingJetEta,
                          double ScoutingJetPhi,
                          const std::vector<Run3ScoutingMuon>& scoutingMuons,
                          double dr2Cut);
  bool isBarrel(double eta);
  bool isEndCapP(double eta);
  bool isEndCapM(double eta);
  bool isForward(double eta);

  void bookMESub(DQMStore::IBooker&,
                 ObjME* a_me,
                 const int len_,
                 const std::string& h_Name,
                 const std::string& h_Title,
                 const std::string& h_subOptName,
                 const std::string& h_subOptTitle,
                 const bool doPhi = true,
                 const bool doEta = true,
                 const bool doEtaPhi = true,
                 const bool doVsLS = true);
  void FillME(ObjME* a_me,
              const double pt_,
              const double phi_,
              const double eta_,
              const int ls_,
              const std::string& denu,
              const bool doPhi = true,
              const bool doEta = true,
              const bool doEtaPhi = true,
              const bool doVsLS = true);

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  double ptcut_;
  bool isPFJetTrig;
  bool isCaloJetTrig;
  bool isPuppiJet;
  bool isScoutingPFJetTrig;
  double dr2cut_;
  bool doVariableBinning;

  int verbose_;
  std::string JetIDQuality_;
  std::string JetIDVersion_;
  Run3ScoutingPFJetIDSelectionFunctor::Quality_t run3scoutingpfjetidquality;
  Run3ScoutingPFJetIDSelectionFunctor::Version_t run3scoutingpfjetidversion;
  Run3ScoutingPFJetIDSelectionFunctor run3scoutingpfjetIDFunctor;

  const bool enableFullMonitoring_;

  edm::InputTag muoInputTag_;
  edm::InputTag vtxInputTag_;
  edm::InputTag scoutingMuonInputTag_;

  const edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  const edm::EDGetTokenT<reco::PFJetCollection> jetSrc_;
  const edm::EDGetTokenT<reco::JetCorrector> correctorToken_;
  const edm::EDGetTokenT<reco::CaloJetCollection> calojetToken_;

  edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> scoutingMuonToken_;
  edm::EDGetTokenT<std::vector<Run3ScoutingPFJet>> scoutjetSrc_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::Muon, true> muoSelection_;

  unsigned nmuons_;

  std::vector<double> jetPt_variable_binning_;

  MEbinning jetpt_binning_;
  MEbinning jetptThr_binning_;
  MEbinning ls_binning_;

  ObjME a_ME[7];
  ObjME a_ME_HB[7];
  ObjME a_ME_HE[7];
  ObjME a_ME_HF[7];
  ObjME a_ME_HE_p[7];
  ObjME a_ME_HE_m[7];

  struct correctedScoutingJets {
    double pt;
    double eta;
    double phi;
  };
  std::vector<correctedScoutingJets> corrected_scoutingjets;

  struct correctedCaloJets {
    double pt;
    double eta;
    double phi;
  };
  std::vector<correctedCaloJets> corrected_calojets;

  // (mia) not optimal, we should make use of variable binning which reflects the detector !
  MEbinning jet_phi_binning_{64, -3.2, 3.2};
  MEbinning jet_eta_binning_{50, -5, 5};
};

JetMonitor::JetMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      ptcut_(iConfig.getParameter<double>("ptcut")),
      isPFJetTrig(iConfig.getParameter<bool>("ispfjettrg")),
      isCaloJetTrig(iConfig.getParameter<bool>("iscalojettrg")),
      isPuppiJet(iConfig.getParameter<bool>("ispuppijet")),
      isScoutingPFJetTrig(iConfig.getParameter<bool>("isscoutingpfjettrg")),
      dr2cut_(iConfig.getParameter<double>("dr2cut")),
      doVariableBinning(iConfig.getParameter<bool>("doVariablebinning")),
      JetIDQuality_(iConfig.getParameter<std::string>("JetIDQuality")),
      JetIDVersion_(iConfig.getParameter<std::string>("JetIDVersion")),
      enableFullMonitoring_(iConfig.getParameter<bool>("enableFullMonitoring")),
      muoInputTag_(iConfig.getParameter<edm::InputTag>("muons")),
      vtxInputTag_(iConfig.getParameter<edm::InputTag>("vertices")),
      scoutingMuonInputTag_(iConfig.getParameter<edm::InputTag>("muons")),
      muoToken_(mayConsume<reco::MuonCollection>(muoInputTag_)),
      vtxToken_(mayConsume<reco::VertexCollection>(vtxInputTag_)),
      jetSrc_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jetSrc"))),
      correctorToken_(mayConsume<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("corrector"))),
      calojetToken_(mayConsume<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("jetSrc"))),
      scoutingMuonToken_(mayConsume<std::vector<Run3ScoutingMuon>>(scoutingMuonInputTag_)),
      scoutjetSrc_(mayConsume<std::vector<Run3ScoutingPFJet>>(iConfig.getParameter<edm::InputTag>("jetSrc"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      muoSelection_(iConfig.getParameter<std::string>("muoSelection")),
      nmuons_(iConfig.getParameter<unsigned>("nmuons")),
      jetPt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double>>("jetptBinning")),
      jetpt_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPSet"))),
      jetptThr_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("jetPtThrPSet"))),
      ls_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))) {}

JetMonitor::~JetMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void JetMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on()) {
    num_genTriggerEventFlag_->initRun(iRun, iSetup);
  }
  if (den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on()) {
    den_genTriggerEventFlag_->initRun(iRun, iSetup);
  }

  // check if every HLT path specified in numerator and denominator has a valid match in the HLT Menu
  hltPathsAreValid_ = (num_genTriggerEventFlag_ && den_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() &&
                       den_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->allHLTPathsAreValid() &&
                       den_genTriggerEventFlag_->allHLTPathsAreValid());

  // if valid HLT paths are required,
  // create DQM outputs only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  std::string histname, histtitle;
  std::string hist_obtag = "";
  std::string histtitle_obtag = "";
  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  if (isPFJetTrig) {   // flag for the trigger path
    if (isPuppiJet) {  // flag for the offline collection
      hist_obtag = "pfpuppijet";
      histtitle_obtag = "PFPuppi Jet";
    } else if (!isPuppiJet) {
      hist_obtag = "pfjet";
      histtitle_obtag = "PFJet";
    }
  } else if (isCaloJetTrig) {
    hist_obtag = "calojet";
    histtitle_obtag = "CaloJet";
  } else if (isScoutingPFJetTrig) {
    hist_obtag = "scoutingpfjet";
    histtitle_obtag = "ScoutingPfJet";
  } else {
    hist_obtag = "pfpuppijet";
    histtitle_obtag = "PFPuppi Jet";
  }  //default is pfpuppijet

  bookMESub(ibooker, a_ME, sizeof(a_ME) / sizeof(a_ME[0]), hist_obtag, histtitle_obtag, "", "");
  bookMESub(ibooker,
            a_ME_HB,
            sizeof(a_ME_HB) / sizeof(a_ME_HB[0]),
            hist_obtag,
            histtitle_obtag,
            "HB",
            "(HB)",
            true,
            true,
            true,
            false);
  bookMESub(ibooker,
            a_ME_HE,
            sizeof(a_ME_HE) / sizeof(a_ME_HE[0]),
            hist_obtag,
            histtitle_obtag,
            "HE",
            "(HE)",
            true,
            true,
            true,
            false);
  bookMESub(ibooker,
            a_ME_HF,
            sizeof(a_ME_HF) / sizeof(a_ME_HF[0]),
            hist_obtag,
            histtitle_obtag,
            "HF",
            "(HF)",
            true,
            true,
            true,
            false);

  //check the flag
  if (!enableFullMonitoring_) {
    return;
  }

  bookMESub(ibooker,
            a_ME_HE_p,
            sizeof(a_ME_HE_p) / sizeof(a_ME_HE_p[0]),
            hist_obtag,
            histtitle_obtag,
            "HE_p",
            "(HE+)",
            true,
            false,
            true,
            false);
  bookMESub(ibooker,
            a_ME_HE_m,
            sizeof(a_ME_HE_m) / sizeof(a_ME_HE_m[0]),
            hist_obtag,
            histtitle_obtag,
            "HE_m",
            "(HE-)",
            true,
            false,
            true,
            false);
}

void JetMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;
  if (!num_genTriggerEventFlag_->on())
    return;

  const int ls = iEvent.id().luminosityBlock();

  //--------- access vrtx -----------
  reco::Vertex vtx;
  edm::Handle<reco::VertexCollection> vtxHandle;
  if (!isScoutingPFJetTrig) {
    iEvent.getByToken(vtxToken_, vtxHandle);
    if (vtxHandle.isValid()) {
      for (auto const& v : *vtxHandle) {
        bool isFake = v.isFake();

        if (!isFake) {
          vtx = v;
          break;
        }
      }
    } else {
      if (vtxInputTag_.label().empty())
        edm::LogWarning("JetMonitor") << "VertexCollection is not set";
      else
        edm::LogWarning("JetMonitor") << "skipping events because the collection " << vtxInputTag_.label().c_str()
                                      << " is not available";
      if (!vtxInputTag_.label().empty())
        return;
    }
  }
  // -------- muons ----------
  std::vector<reco::Muon> muons;
  edm::Handle<reco::MuonCollection> muoHandle;

  std::vector<Run3ScoutingMuon> scoutingmuons;
  edm::Handle<std::vector<Run3ScoutingMuon>> ScoutingMuonHandle;

  if (isScoutingPFJetTrig) {
    iEvent.getByToken(scoutingMuonToken_, ScoutingMuonHandle);
    if (ScoutingMuonHandle.isValid()) {
      if (ScoutingMuonHandle->size() < nmuons_) {
        //edm::LogWarning("JetMonitor") << "Run3ScoutingMuon collection not valid.";
        return;
      }
      for (auto const& iscoutmuon : *ScoutingMuonHandle) {
        //std::cout << "scouting muon pT: " << iscoutmuon.pt() << std::endl;
        //std::cout << "scouting muon eta: " << iscoutmuon.eta() << std::endl;
        if (isGoodScoutingMuon(iscoutmuon)) {
          scoutingmuons.push_back(iscoutmuon);
        }
      }
      if (scoutingmuons.size() < nmuons_) {  // require 1 tight scouting muon if orthogonal method, else nmuons_ is 0
        return;
      }
    } else {
      if (scoutingMuonInputTag_.label().empty()) {
        edm::LogWarning("JetMonitor") << "Scouting muon collection not valid \n";
      } else {
        edm::LogWarning("JetMonitor") << "skipping events because the collection "
                                      << scoutingMuonInputTag_.label().c_str() << " is not available \n";
      }
      return;
    }
  } else {
    iEvent.getByToken(muoToken_, muoHandle);
    if (muoHandle.isValid()) {
      if (muoHandle->size() < nmuons_)
        return;
      for (auto const& m : *muoHandle) {
        bool istightID = m.isGlobalMuon() && m.isPFMuon() && m.globalTrack()->normalizedChi2() < 10. &&
                         m.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 && m.numberOfMatchedStations() > 1 &&
                         fabs(m.muonBestTrack()->dxy(vtx.position())) < 0.2 &&
                         fabs(m.muonBestTrack()->dz(vtx.position())) < 0.5 &&
                         m.innerTrack()->hitPattern().numberOfValidPixelHits() > 0 &&
                         m.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5;
        if (muoSelection_(m) && istightID) {
          muons.push_back(m);
        }
      }
      if (muons.size() < nmuons_)  // require 1 tight muon if orthogonal method, else nmuons_ is 0
        return;
    } else {
      if (muoInputTag_.label().empty())
        edm::LogWarning("JetMonitor") << "MuonCollection not set";
      else
        edm::LogWarning("JetMonitor") << "skipping events because the collection " << muoInputTag_.label().c_str()
                                      << " is not available";
      if (!muoInputTag_.label().empty())
        return;
    }
  }
  // ------------- Jets ------------
  corrected_jets.clear();
  corrected_calojets.clear();
  corrected_scoutingjets.clear();

  edm::Handle<reco::PFJetCollection> PFjetHandle;
  edm::Handle<reco::CaloJetCollection> calojetHandle;
  edm::Handle<std::vector<Run3ScoutingPFJet>> ScoutingJetHandle;

  edm::Handle<reco::JetCorrector> Corrector;
  iEvent.getByToken(correctorToken_, Corrector);

  if (isPFJetTrig) {  // if pfjet
    iEvent.getByToken(jetSrc_, PFjetHandle);
    if (!PFjetHandle.isValid()) {
      edm::LogWarning("JetMonitor") << "Jet handle not valid \n";
      return;
    }

    for (auto const& ijet : *PFjetHandle) {
      // Clean Jets
      if (!isCleanJet(ijet.eta(), ijet.phi(), muons, dr2cut_))
        continue;

      // apply corrections on the fly
      double jec = Corrector.isValid() ? Corrector->correction(ijet) : 1.0;
      double corjet = jec * ijet.pt();
      if (corjet < ptcut_) {
        continue;
      }
      corrected_jets.push_back({corjet,
                                ijet.eta(),
                                ijet.phi(),
                                ijet.neutralHadronEnergyFraction(),
                                ijet.neutralEmEnergyFraction(),
                                ijet.chargedHadronEnergyFraction(),
                                ijet.chargedEmEnergyFraction(),
                                ijet.muonEnergyFraction(),
                                ijet.neutralMultiplicity(),
                                ijet.chargedMultiplicity()});

    }  // end for jets
    std::sort(corrected_jets.begin(), corrected_jets.end(), [](const auto& a, const auto& b) { return a.pt > b.pt; });
    if (corrected_jets.empty())
      return;
    if (!passTightJetID(corrected_jets[0]))
      return;

  }  // end if PF Jets

  if (isCaloJetTrig) {  //if calojet
    iEvent.getByToken(calojetToken_, calojetHandle);
    if (!calojetHandle.isValid()) {
      edm::LogWarning("JetMonitor") << "Jet handle not valid \n";
      return;
    }
    for (auto const& j : *calojetHandle) {
      // Clean Jets
      if (!isCleanJet(j.eta(), j.phi(), muons, dr2cut_))
        continue;

      // apply corrections on the fly
      double jec = Corrector.isValid() ? Corrector->correction(j) : 1.0;
      double corjet = jec * j.pt();
      if (corjet < ptcut_) {
        continue;
      }
      corrected_calojets.push_back({corjet, j.eta(), j.phi()});
    }  // end for jets
    std::sort(
        corrected_calojets.begin(), corrected_calojets.end(), [](const auto& a, const auto& b) { return a.pt > b.pt; });
    if (corrected_calojets.empty())
      return;

  }  // end if Calo Jets

  if (isScoutingPFJetTrig) {  //if scouting pf jets
    iEvent.getByToken(scoutjetSrc_, ScoutingJetHandle);
    if (!ScoutingJetHandle.isValid()) {
      edm::LogWarning("JetMonitor") << "Scouting jet handle not valid \n";
      return;
    }

    if (JetIDVersion_ == "RUN3Scouting") {
      run3scoutingpfjetidversion = Run3ScoutingPFJetIDSelectionFunctor::RUN3Scouting;
    } else {
      if (verbose_)
        std::cout << "no valid scouting Run3ScoutinPF JetID version given" << std::endl;
    }
    if (JetIDQuality_ == "TIGHT") {
      run3scoutingpfjetidquality = Run3ScoutingPFJetIDSelectionFunctor::TIGHT;
    } else if (JetIDQuality_ == "TIGHTLEPVETO") {
      run3scoutingpfjetidquality = Run3ScoutingPFJetIDSelectionFunctor::TIGHTLEPVETO;
    } else {
      if (verbose_)
        std::cout << "no Valid scouting Run3ScoutinPF JetID quality given" << std::endl;
    }
    run3scoutingpfjetIDFunctor =
        Run3ScoutingPFJetIDSelectionFunctor(run3scoutingpfjetidversion, run3scoutingpfjetidquality);
    for (auto const& iscoutjet : *ScoutingJetHandle) {
      bool passScoutjetID = false;
      passScoutjetID = run3scoutingpfjetIDFunctor(iscoutjet);
      if (!passScoutjetID)
        continue;
      if (!isCleanScoutingJet(iscoutjet.eta(), iscoutjet.phi(), scoutingmuons, dr2cut_))
        continue;

      reco::PFJet dummy_scoutingpfjet;
      reco::Particle::PolarLorentzVector dummy_scoutingpfjetP4(
          iscoutjet.pt(), iscoutjet.eta(), iscoutjet.phi(), iscoutjet.m());
      dummy_scoutingpfjet.setP4(dummy_scoutingpfjetP4);
      dummy_scoutingpfjet.setJetArea(iscoutjet.jetArea());

      // apply corrections on the fly
      double jec = Corrector.isValid() ? Corrector->correction(dummy_scoutingpfjet) : 1.0;
      //std::cout << "Jet Pt: "<< iscoutjet.pt() << " JEC: " << jec << std::endl;

      double corjet = jec * iscoutjet.pt();  /////------> iscoutjet or dummy_...jet ?????
      if (corjet < ptcut_) {
        continue;
      }
      corrected_scoutingjets.push_back({corjet, iscoutjet.eta(), iscoutjet.phi()});
    }  // end for loop over jets
    std::sort(corrected_scoutingjets.begin(), corrected_scoutingjets.end(), [](const auto& a, const auto& b) {
      return a.pt > b.pt;
    });
    if (corrected_scoutingjets.empty())
      return;
  }  //end if scouting pf jets

  double jetpt_ = (isPFJetTrig && !corrected_jets.empty())                   ? corrected_jets[0].pt
                  : (isCaloJetTrig && !corrected_calojets.empty())           ? corrected_calojets[0].pt
                  : (isScoutingPFJetTrig && !corrected_scoutingjets.empty()) ? corrected_scoutingjets[0].pt
                                                                             : -99.;
  double jeteta_ = (isPFJetTrig && !corrected_jets.empty())                   ? corrected_jets[0].eta
                   : (isCaloJetTrig && !corrected_calojets.empty())           ? corrected_calojets[0].eta
                   : (isScoutingPFJetTrig && !corrected_scoutingjets.empty()) ? corrected_scoutingjets[0].eta
                                                                              : 99;
  double jetphi_ = (isPFJetTrig && !corrected_jets.empty())                   ? corrected_jets[0].phi
                   : (isCaloJetTrig && !corrected_calojets.empty())           ? corrected_calojets[0].phi
                   : (isScoutingPFJetTrig && !corrected_scoutingjets.empty()) ? corrected_scoutingjets[0].phi
                                                                              : 99;

  FillME(a_ME, jetpt_, jetphi_, jeteta_, ls, "denominator");
  if (isBarrel(jeteta_)) {
    FillME(a_ME_HB, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
  } else if (isEndCapP(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
    if (enableFullMonitoring_) {
      FillME(a_ME_HE_p, jetpt_, jetphi_, jeteta_, ls, "denominator", true, false, true, false);
    }
  } else if (isEndCapM(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
    if (enableFullMonitoring_) {
      FillME(a_ME_HE_m, jetpt_, jetphi_, jeteta_, ls, "denominator", true, false, true, false);
    }
  } else if (isForward(jeteta_)) {
    FillME(a_ME_HF, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
  }
  // Require Numerator //
  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  FillME(a_ME, jetpt_, jetphi_, jeteta_, ls, "numerator");
  if (isBarrel(jeteta_)) {
    FillME(a_ME_HB, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
  } else if (isEndCapP(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
    if (enableFullMonitoring_) {
      FillME(a_ME_HE_p, jetpt_, jetphi_, jeteta_, ls, "numerator", true, false, true, false);
    }
  } else if (isEndCapM(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
    if (enableFullMonitoring_) {
      FillME(a_ME_HE_m, jetpt_, jetphi_, jeteta_, ls, "numerator", true, false, true, false);
    }
  } else if (isForward(jeteta_)) {
    FillME(a_ME_HF, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
  }
}

bool JetMonitor::passTightJetID(const correctedPFJets& jet) {
  const double abseta = std::abs(jet.eta);

  const double NHF = jet.NHF;
  const double NEMF = jet.NEMF;
  const double CHF = jet.CHF;
  const double CEMF = jet.CEMF;
  const double MUF = jet.MUF;
  const int NumNeutralParticles = jet.NumNeutralParticles;
  const int CHM = jet.CHM;
  const int NumConst = CHM + NumNeutralParticles;
  bool passjetID = false;
  // Id for puppi jets
  if (abseta <= 2.6) {
    passjetID = (CEMF < 0.8 && CHM > 0 && CHF > 0.01 && NumConst > 1 && NEMF < 0.9 && MUF < 0.8 && NHF < 0.9);
  } else if (abseta > 2.6 && abseta <= 2.7) {
    passjetID = ((CEMF < 0.8 && NEMF < 0.99 && MUF < 0.8 && NHF < 0.9));
  } else if (abseta > 2.7 && abseta <= 3.0) {
    passjetID = (NHF < 0.9999);
  } else if (abseta > 3.0) {
    passjetID = (NEMF < 0.90 && NumNeutralParticles > 2);
  }

  return passjetID;
}

bool JetMonitor::isCleanJet(double JetEta, double JetPhi, const std::vector<reco::Muon>& muons, double dr2Cut) {
  for (const auto& mu : muons) {
    double dR2 = deltaR2(JetEta, JetPhi, mu.eta(), mu.phi());
    if (dR2 < dr2Cut) {
      return false;
    }
  }
  return true;
}

bool JetMonitor::isGoodScoutingMuon(Run3ScoutingMuon const& scoutingMuon) {
  if (scoutingMuon.pt() > 1
      && abs(scoutingMuon.eta()) < 0.8
      && abs(scoutingMuon.trk_dxy()) < 0.2
      /////&& abs(scoutingMuon.trackIso()) < 0.15 //removed
      && abs(scoutingMuon.trk_dz()) < 0.5
      && scoutingMuon.normalizedChi2() < 3
      && scoutingMuon.nValidRecoMuonHits() > 0
      && scoutingMuon.nRecoMuonMatchedStations() > 1
      && scoutingMuon.nValidPixelHits() > 0
      && scoutingMuon.nTrackerLayersWithMeasurement() > 5) {
    return true;
  } else {
    return false;
  }
}

bool JetMonitor::isCleanScoutingJet(double ScoutingJetEta,
                                    double ScoutingJetPhi,
                                    const std::vector<Run3ScoutingMuon>& scoutingMuons,
                                    double dr2Cut) {
  for (const auto& scoutingMuon : scoutingMuons) {
    double ScoutingdR2 = deltaR2(ScoutingJetEta, ScoutingJetPhi, scoutingMuon.eta(), scoutingMuon.phi());
    if (ScoutingdR2 < dr2Cut) {
      return false;
    }
  }
  return true;
}

bool JetMonitor::isBarrel(double eta) {
  bool output = false;
  if (fabs(eta) <= 1.3)
    output = true;
  return output;
}

bool JetMonitor::isEndCapM(double eta) {
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta < 0))
    output = true;  // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}

/// For Hcal Endcap Plus Area
bool JetMonitor::isEndCapP(double eta) {
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0))
    output = true;  // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}

/// For Hcal Forward Area
bool JetMonitor::isForward(double eta) {
  bool output = false;
  if (fabs(eta) > 3.0)
    output = true;
  return output;
}

void JetMonitor::FillME(ObjME* a_me,
                        const double pt_,
                        const double phi_,
                        const double eta_,
                        const int ls_,
                        const std::string& DenoOrNume,
                        const bool doPhi,
                        const bool doEta,
                        const bool doEtaPhi,
                        const bool doVsLS) {
  if (DenoOrNume == "denominator") {
    // index 0 = pt, 1 = ptThreshold , 2 = pt vs ls , 3 = phi, 4 = eta,
    // 5 = eta vs phi, 6 = eta vs pt , 7 = abs(eta) , 8 = abs(eta) vs phi
    a_me[0].denominator->Fill(pt_);  // pt
    a_me[1].denominator->Fill(pt_);  // jetpT Threshold binning for pt
    if (doVsLS)
      a_me[2].denominator->Fill(ls_, pt_);  // pt vs ls
    if (doPhi)
      a_me[3].denominator->Fill(phi_);  // phi
    if (doEta)
      a_me[4].denominator->Fill(eta_);  // eta
    if (doEtaPhi)
      a_me[5].denominator->Fill(eta_, phi_);  // eta vs phi
    if (doEtaPhi)
      a_me[6].denominator->Fill(eta_, pt_);  // eta vs pT
  } else if (DenoOrNume == "numerator") {
    a_me[0].numerator->Fill(pt_);  // pt
    a_me[1].numerator->Fill(pt_);  // jetpT Threshold binning for pt
    if (doVsLS)
      a_me[2].numerator->Fill(ls_, pt_);  // pt vs ls
    if (doPhi)
      a_me[3].numerator->Fill(phi_);  // phi
    if (doEta)
      a_me[4].numerator->Fill(eta_);  // eta
    if (doEtaPhi)
      a_me[5].numerator->Fill(eta_, phi_);  // eta vs phi
    if (doEtaPhi)
      a_me[6].numerator->Fill(eta_, pt_);  // eta vs pT
  } else {
    edm::LogWarning("JetMonitor") << "CHECK OUT denu option in FillME !!! DenoOrNume ? : " << DenoOrNume << std::endl;
  }
}

void JetMonitor::bookMESub(DQMStore::IBooker& Ibooker,
                           ObjME* a_me,
                           const int len_,
                           const std::string& h_Name,
                           const std::string& h_Title,
                           const std::string& h_subOptName,
                           const std::string& hSubT,
                           const bool doPhi,
                           const bool doEta,
                           const bool doEtaPhi,
                           const bool doVsLS) {
  std::string hName = h_Name;
  std::string hTitle = h_Title;
  const std::string hSubN = h_subOptName.empty() ? "" : "_" + h_subOptName;

  int nbin_phi = jet_phi_binning_.nbins;
  double maxbin_phi = jet_phi_binning_.xmax;
  double minbin_phi = jet_phi_binning_.xmin;

  int nbin_eta = jet_eta_binning_.nbins;
  double maxbin_eta = jet_eta_binning_.xmax;
  double minbin_eta = jet_eta_binning_.xmin;

  if (doVariableBinning) {
    hName = h_Name + "pT" + hSubN;
    hTitle = h_Title + " pT " + hSubT;
    bookME(Ibooker, a_me[0], hName, hTitle, jetPt_variable_binning_);
    setMETitle(a_me[0], h_Title + " pT [GeV]", "events / [GeV]");
  } else {
    hName = h_Name + "pT" + hSubN;
    hTitle = h_Title + " pT " + hSubT;
    bookME(Ibooker, a_me[0], hName, hTitle, jetpt_binning_.nbins, jetpt_binning_.xmin, jetpt_binning_.xmax);
    setMETitle(a_me[0], h_Title + " pT [GeV]", "events / [GeV]");
  }

  hName = h_Name + "pT_pTThresh" + hSubN;
  hTitle = h_Title + " pT " + hSubT;
  bookME(Ibooker, a_me[1], hName, hTitle, jetptThr_binning_.nbins, jetptThr_binning_.xmin, jetptThr_binning_.xmax);
  setMETitle(a_me[1], h_Title + "pT [GeV]", "events / [GeV]");

  if (doVsLS) {
    hName = h_Name + "pTVsLS" + hSubN;
    hTitle = h_Title + " vs LS " + hSubT;
    bookME(Ibooker,
           a_me[2],
           hName,
           hTitle,
           ls_binning_.nbins,
           ls_binning_.xmin,
           ls_binning_.xmax,
           jetpt_binning_.xmin,
           jetpt_binning_.xmax);
    setMETitle(a_me[2], "LS", h_Title + "pT [GeV]");
  }

  if (doPhi) {
    hName = h_Name + "phi" + hSubN;
    hTitle = h_Title + " phi " + hSubT;
    bookME(Ibooker, a_me[3], hName, hTitle, nbin_phi, minbin_phi, maxbin_phi);
    setMETitle(a_me[3], h_Title + " #phi", "events / 0.1 rad");
  }

  if (doEta) {
    hName = h_Name + "eta" + hSubN;
    hTitle = h_Title + " eta " + hSubT;
    bookME(Ibooker, a_me[4], hName, hTitle, nbin_eta, minbin_eta, maxbin_eta);
    setMETitle(a_me[4], h_Title + " #eta", "events");
  }

  if (doEtaPhi) {
    hName = h_Name + "EtaVsPhi" + hSubN;
    hTitle = h_Title + " eta Vs phi " + hSubT;
    bookME(Ibooker, a_me[5], hName, hTitle, nbin_eta, minbin_eta, maxbin_eta, nbin_phi, minbin_phi, maxbin_phi);
    setMETitle(a_me[5], h_Title + " #eta", "#phi");
  }

  if (doEtaPhi) {
    hName = h_Name + "EtaVspT" + hSubN;
    hTitle = h_Title + " eta Vs pT " + hSubT;
    bookME(Ibooker,
           a_me[6],
           hName,
           hTitle,
           nbin_eta,
           minbin_eta,
           maxbin_eta,
           jetpt_binning_.nbins,
           jetpt_binning_.xmin,
           jetpt_binning_.xmax);
    setMETitle(a_me[6], h_Title + " #eta", "Leading Jet pT [GeV]");
  }
}

void JetMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/Jet");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJetsPuppi"));
  desc.add<edm::InputTag>("corrector", edm::InputTag(""));
  desc.add<double>("ptcut", 30);
  desc.add<bool>("ispfjettrg", true);
  desc.add<bool>("iscalojettrg", false);
  desc.add<bool>("ispuppijet", false);
  desc.add<double>("dr2cut", 0.16);

  desc.add<bool>("isscoutingpfjettrg", false);
  desc.add<bool>("doVariablebinning", false);

  desc.add<std::string>("JetIDQuality", "TIGHT");
  desc.add<std::string>("JetIDVersion", "RUN3Scouting");

  desc.add<bool>("enableFullMonitoring", true);
  desc.add<std::string>("muoSelection", "pt > 30");
  desc.add<unsigned>("nmuons", 0);

  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription jetPSet;
  edm::ParameterSetDescription jetPtThrPSet;
  fillHistoPSetDescription(jetPSet);
  fillHistoPSetDescription(jetPtThrPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPSet", jetPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPtThrPSet", jetPtThrPSet);
  histoPSet.add<std::vector<double>>("jetptBinning",
                                     {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                                      170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.});

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("jetMonitoring", desc);
}

DEFINE_FWK_MODULE(JetMonitor);

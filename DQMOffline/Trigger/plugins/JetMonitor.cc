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

  bool isBarrel(double eta);
  bool isEndCapP(double eta);
  bool isEndCapM(double eta);
  bool isForward(double eta);
  bool isHEP17(double eta, double phi);
  bool isHEM17(double eta, double phi);
  bool isHEP18(double eta, double phi);  // -0.87< Phi < -1.22

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

  edm::EDGetTokenT<edm::View<reco::Jet> > jetSrc_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  MEbinning jetpt_binning_;
  MEbinning jetptThr_binning_;
  MEbinning ls_binning_;

  ObjME a_ME[7];
  ObjME a_ME_HB[7];
  ObjME a_ME_HE[7];
  ObjME a_ME_HF[7];
  ObjME a_ME_HE_p[7];
  ObjME a_ME_HE_m[7];
  ObjME a_ME_HEM17[7];
  ObjME a_ME_HEP17[7];
  ObjME a_ME_HEP18[7];

  ObjME jetHEP17_AbsEtaVsPhi_;
  ObjME jetHEM17_AbsEtaVsPhi_;
  ObjME jetHEP17_AbsEta_;
  ObjME jetHEM17_AbsEta_;

  std::vector<double> v_jetpt;
  std::vector<double> v_jeteta;
  std::vector<double> v_jetphi;

  // (mia) not optimal, we should make use of variable binning which reflects the detector !
  MEbinning jet_phi_binning_{32, -3.2, 3.2};
  MEbinning jet_eta_binning_{20, -5, 5};

  MEbinning eta_binning_hep17_{9, 1.3, 3.0};
  MEbinning eta_binning_hem17_{9, -3.0, -1.3};

  MEbinning phi_binning_hep17_{7, -0.87, -0.52};
  MEbinning phi_binning_hep18_{7, -0.52, -0.17};
};

JetMonitor::JetMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      ptcut_(iConfig.getParameter<double>("ptcut")),
      isPFJetTrig(iConfig.getParameter<bool>("ispfjettrg")),
      isCaloJetTrig(iConfig.getParameter<bool>("iscalojettrg")),
      jetSrc_(mayConsume<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jetSrc"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
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

  if (isPFJetTrig) {
    hist_obtag = "pfjet";
    histtitle_obtag = "PFJet";
  } else if (isCaloJetTrig) {
    hist_obtag = "calojet";
    histtitle_obtag = "CaloJet";
  } else {
    hist_obtag = "pfjet";
    histtitle_obtag = "PFJet";
  }  //default is pfjet

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
  bookMESub(ibooker,
            a_ME_HE_p,
            sizeof(a_ME_HE_p) / sizeof(a_ME_HE_p[0]),
            hist_obtag,
            histtitle_obtag,
            "HE_p",
            "(HE+)",
            true,
            true,
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
            true,
            true,
            false);
  bookMESub(ibooker,
            a_ME_HEP17,
            sizeof(a_ME_HEP17) / sizeof(a_ME_HEP17[0]),
            hist_obtag,
            histtitle_obtag,
            "HEP17",
            "(HEP17)",
            true,
            false,
            false,
            false);
  bookMESub(ibooker,
            a_ME_HEM17,
            sizeof(a_ME_HEM17) / sizeof(a_ME_HEM17[0]),
            hist_obtag,
            histtitle_obtag,
            "HEM17",
            "(HEM17)",
            true,
            false,
            false,
            false);
  bookMESub(ibooker,
            a_ME_HEP18,
            sizeof(a_ME_HEP18) / sizeof(a_ME_HEP18[0]),
            hist_obtag,
            histtitle_obtag,
            "HEP18",
            "(HEP18)",
            false,
            false,
            false,
            false);

  /*
    WE WOULD NEED TURNON CURVES TO BE COMPARED NOT JUST THE ZOOM OF A 2D MAP !!!

  histname = hist_obtag +"AbsEtaVsPhi_HEP17"; histtitle = histtitle_obtag + " |eta| Vs phi (HEP17) ";
  bookME(ibooker,jetHEP17_AbsEtaVsPhi_,histname,histtitle, eta_binning_hep17_.nbins, eta_binning_hep17_.xmin, eta_binning_hep17_.xmax, phi_binning_hep17_.nbins,phi_binning_hep17_.xmin,phi_binning_hep17_.xmax);
  setMETitle(jetHEP17_AbsEtaVsPhi_,histtitle_obtag + " |#eta|","#phi");

  histname = hist_obtag +"AbsEtaVsPhi_HEM17"; histtitle = histtitle_obtag + " |eta| Vs phi (HEM17) ";
  bookME(ibooker,jetHEM17_AbsEtaVsPhi_,histname,histtitle, eta_binning_hep17_.nbins, eta_binning_hep17_.xmin, eta_binning_hep17_.xmax, phi_binning_hep17_.nbins,phi_binning_hep17_.xmin,phi_binning_hep17_.xmax);
  setMETitle(jetHEM17_AbsEtaVsPhi_,histtitle_obtag + " |#eta|","#phi");
  */

  histname = hist_obtag + "abseta_HEP17";
  histtitle = histtitle_obtag + " |#eta| (HEP17) ";
  bookME(ibooker,
         jetHEP17_AbsEta_,
         histname,
         histtitle,
         eta_binning_hep17_.nbins,
         eta_binning_hep17_.xmin,
         eta_binning_hep17_.xmax);
  setMETitle(jetHEP17_AbsEta_, histtitle_obtag + " |#eta|", "events / |#eta|");

  histname = hist_obtag + "abseta_HEM17";
  histtitle = histtitle_obtag + " |eta| (HEM17) ";
  bookME(ibooker,
         jetHEM17_AbsEta_,
         histname,
         histtitle,
         eta_binning_hep17_.nbins,
         eta_binning_hep17_.xmin,
         eta_binning_hep17_.xmax);
  setMETitle(jetHEM17_AbsEta_, histtitle_obtag + " |#eta|", "events / |#eta|");
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

  const int ls = iEvent.id().luminosityBlock();

  v_jetpt.clear();
  v_jeteta.clear();
  v_jetphi.clear();

  edm::Handle<edm::View<reco::Jet> > offjets;
  iEvent.getByToken(jetSrc_, offjets);
  if (!offjets.isValid()) {
    edm::LogWarning("JetMonitor") << "Jet handle not valid \n";
    return;
  }
  for (edm::View<reco::Jet>::const_iterator ibegin = offjets->begin(), iend = offjets->end(), ijet = ibegin;
       ijet != iend;
       ++ijet) {
    //if (ijet->pt()< 20) {continue;}
    if (ijet->pt() < ptcut_) {
      continue;
    }
    v_jetpt.push_back(ijet->pt());
    v_jeteta.push_back(ijet->eta());
    v_jetphi.push_back(ijet->phi());
    //    cout << "jetpt (view ) : " << ijet->pt() << endl;
  }

  if (v_jetpt.empty())
    return;
  double jetpt_ = v_jetpt[0];
  double jeteta_ = v_jeteta[0];
  double jetphi_ = v_jetphi[0];

  FillME(a_ME, jetpt_, jetphi_, jeteta_, ls, "denominator");
  if (isBarrel(jeteta_)) {
    FillME(a_ME_HB, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
  } else if (isEndCapP(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
    FillME(a_ME_HE_p, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
  } else if (isEndCapM(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
    FillME(a_ME_HE_m, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
  } else if (isForward(jeteta_)) {
    FillME(a_ME_HF, jetpt_, jetphi_, jeteta_, ls, "denominator", true, true, true, false);
  }

  if (isHEP17(jeteta_, jetphi_)) {
    FillME(a_ME_HEP17,
           jetpt_,
           jetphi_,
           jeteta_,
           ls,
           "denominator",
           true,
           false,
           false,
           false);  // doPhi, doEta, doEtaPhi, doVsLS
    jetHEP17_AbsEta_.denominator->Fill(abs(jeteta_));
  } else if (isHEM17(jeteta_, jetphi_)) {
    FillME(
        a_ME_HEM17, jetpt_, jetphi_, jeteta_, ls, "denominator", true, false, false, false);  // doPhi, doEta, doEtaPhi
    jetHEM17_AbsEta_.denominator->Fill(abs(jeteta_));
  } else if (isHEP18(jeteta_, jetphi_)) {
    FillME(
        a_ME_HEP18, jetpt_, jetphi_, jeteta_, ls, "denominator", false, false, false, false);  // doPhi, doEta, doEtaPhi
  }

  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;  // Require Numerator //

  FillME(a_ME, jetpt_, jetphi_, jeteta_, ls, "numerator");
  if (isBarrel(jeteta_)) {
    FillME(a_ME_HB, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
  } else if (isEndCapP(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
    FillME(a_ME_HE_p, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
  } else if (isEndCapM(jeteta_)) {
    FillME(a_ME_HE, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
    FillME(a_ME_HE_m, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
  } else if (isForward(jeteta_)) {
    FillME(a_ME_HF, jetpt_, jetphi_, jeteta_, ls, "numerator", true, true, true, false);
  }

  if (isHEP17(jeteta_, jetphi_)) {
    FillME(a_ME_HEP17,
           jetpt_,
           jetphi_,
           jeteta_,
           ls,
           "numerator",
           true,
           false,
           false,
           false);  // doPhi, doEta, doEtaPhi, doVsLS
    jetHEP17_AbsEta_.numerator->Fill(abs(jeteta_));
  } else if (isHEM17(jeteta_, jetphi_)) {
    FillME(a_ME_HEM17,
           jetpt_,
           jetphi_,
           jeteta_,
           ls,
           "numerator",
           true,
           false,
           false,
           false);  // doPhi, doEta, doEtaPhi, doVsLS
    jetHEM17_AbsEta_.numerator->Fill(abs(jeteta_));
  } else if (isHEP18(jeteta_, jetphi_)) {
    FillME(a_ME_HEP18,
           jetpt_,
           jetphi_,
           jeteta_,
           ls,
           "numerator",
           false,
           false,
           false,
           false);  // doPhi, doEta, doEtaPhi, doVsLS
  }
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
  //if ( eta<=3.0 && eta >1.3) output=true;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0))
    output = true;  // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}

/// For Hcal Forward Plus Area
bool JetMonitor::isForward(double eta) {
  bool output = false;
  if (fabs(eta) > 3.0)
    output = true;
  return output;
}

/// For Hcal HEP17 Area
bool JetMonitor::isHEP17(double eta, double phi) {
  bool output = false;
  // phi -0.87 to -0.52
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0) && phi > -0.87 && phi <= -0.52) {
    output = true;
  }  // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}

/// For Hcal HEM17 Area
bool JetMonitor::isHEM17(double eta, double phi) {
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta < 0) && phi > -0.87 && phi <= -0.52) {
    output = true;
  }  // (mia) this magic number should come from some file in CMSSW !!!
  return output;
}

/// For Hcal HEP18 Area
bool JetMonitor::isHEP18(double eta, double phi) {
  bool output = false;
  // phi -0.87 to -0.52
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta > 0) && phi > -0.52 && phi <= -0.17) {
    output = true;
  }  // (mia) this magic number should come from some file in CMSSW !!!
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
    if (doEta)
      a_me[6].denominator->Fill(eta_, pt_);  // eta vs pT
  } else if (DenoOrNume == "numerator") {
    a_me[0].numerator->Fill(pt_);  // pt
    a_me[1].numerator->Fill(pt_);  // jetpT Threshold binning for pt
    if (doVsLS)
      a_me[2].numerator->Fill(ls_, pt_);  // pt vs ls
    if (doPhi)
      a_me[3].numerator->Fill(phi_);  // phi
    if (doEta)
      a_me[4].numerator->Fill(eta_);  // eat
    if (doEtaPhi)
      a_me[5].numerator->Fill(eta_, phi_);  // eta vs phi
    if (doEta)
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

  if (h_subOptName == "HEP17") {
    nbin_phi = phi_binning_hep17_.nbins;
    maxbin_phi = phi_binning_hep17_.xmax;
    minbin_phi = phi_binning_hep17_.xmin;

    nbin_eta = eta_binning_hep17_.nbins;
    maxbin_eta = eta_binning_hep17_.xmax;
    minbin_eta = eta_binning_hep17_.xmin;
  }
  if (h_subOptName == "HEM17") {
    nbin_phi = phi_binning_hep17_.nbins;
    maxbin_phi = phi_binning_hep17_.xmax;
    minbin_phi = phi_binning_hep17_.xmin;

    nbin_eta = eta_binning_hem17_.nbins;
    maxbin_eta = eta_binning_hem17_.xmax;
    minbin_eta = eta_binning_hem17_.xmin;
  }
  if (h_subOptName == "HEP18") {
    nbin_phi = phi_binning_hep18_.nbins;
    maxbin_phi = phi_binning_hep18_.xmax;
    minbin_phi = phi_binning_hep18_.xmin;

    nbin_eta = eta_binning_hep17_.nbins;
    maxbin_eta = eta_binning_hep17_.xmax;
    minbin_eta = eta_binning_hep17_.xmin;
  }
  hName = h_Name + "pT" + hSubN;
  hTitle = h_Title + " pT " + hSubT;
  bookME(Ibooker, a_me[0], hName, hTitle, jetpt_binning_.nbins, jetpt_binning_.xmin, jetpt_binning_.xmax);
  setMETitle(a_me[0], h_Title + " pT [GeV]", "events / [GeV]");

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
    setMETitle(a_me[4], h_Title + " #eta", "events / #eta");
  }

  if (doEtaPhi) {
    hName = h_Name + "EtaVsPhi" + hSubN;
    hTitle = h_Title + " eta Vs phi " + hSubT;
    bookME(Ibooker, a_me[5], hName, hTitle, nbin_eta, minbin_eta, maxbin_eta, nbin_phi, minbin_phi, maxbin_phi);
    setMETitle(a_me[5], h_Title + " #eta", "#phi");
  }

  if (doEta) {
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

  desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJetsCHS"));
  desc.add<double>("ptcut", 20);
  desc.add<bool>("ispfjettrg", true);
  desc.add<bool>("iscalojettrg", false);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi"));
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions", {});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel", "");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT"));
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths", {});
  genericTriggerEventPSet.add<std::string>("hltDBKey", "");
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription jetPSet;
  edm::ParameterSetDescription jetPtThrPSet;
  fillHistoPSetDescription(jetPSet);
  fillHistoPSetDescription(jetPtThrPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPSet", jetPSet);
  histoPSet.add<edm::ParameterSetDescription>("jetPtThrPSet", jetPtThrPSet);
  histoPSet.add<std::vector<double> >("jetptBinning",
                                      {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                                       170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.});

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("jetMonitoring", desc);
}

DEFINE_FWK_MODULE(JetMonitor);

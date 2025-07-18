#include <string>
#include <vector>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include "TLorentzVector.h"
#include <cassert>
#include "TPRegexp.h"

class ZGammaplusJetsMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  ZGammaplusJetsMonitor(const edm::ParameterSet&);
  ~ZGammaplusJetsMonitor() throw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;

  bool islowRefPt(double RefPt);
  bool ismediumRefPt(double RefPt);
  bool ishighRefPt(double RefPt);

  bool isBarrel(double eta);
  bool isEndCapInner(double eta);
  bool isEndCapOuter(double eta);
  bool isForward(double eta);

  bool isMatched(double hltJetEta, double hltJetPhi, double OffJetEta, double OffJetPhi);

  void bookMESub(DQMStore::IBooker&,
                 ObjME* a_me,
                 const int len_,
                 const std::string& h_Name,
                 const std::string& h_Title,
                 const std::string& h_subOptName,
                 const std::string& h_subOptTitle,
                 const bool doDirectBalancevsReferencePt = true,
                 const bool bookDen = false);

  void fillME(ObjME* a_me,
              const double directbalance_,
              const double Difjetref_,
              const double Assymetry_,
              const double ReferencePt_,
              const double Jetpt_,
              const bool doDirectBalancevsReferencePt = true);

private:
  const std::string folderName_;

  const std::string processName_;  // process name of (HLT) process for which to get HLT configuration
  // The instance of the HLTConfigProvider as a data member
  HLTConfigProvider hltConfig_;

  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventObject_;
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const std::string pathName;
  const std::string moduleName;
  const edm::InputTag jetInputTag_;
  const edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  const edm::EDGetTokenT<reco::JetCorrector> correctorToken_;

  double muon_pt;
  double muon_eta;
  double pt_cut;
  double Z_DM;
  double Z_Pt;
  double dphi_cut;
  double offline_cut;
  bool isMuonPath_;

  std::vector<double> directbalance_Binning;
  std::vector<double> TrObjPt_Binning;
  std::vector<double> jetpt_Binning;
  MEbinning DifJetRefPT_Binning{50, -75.0, 75.0};

  //# of variables: 1. direct balance, 2. difference, 3. asymmetry, 4. hltobjectPt, 5. hltJetPt, 6. directbalanceVShltobjectPt
  ObjME a_ME[6];
  ObjME a_ME_HB[6];
  ObjME a_ME_HE_I[6];
  ObjME a_ME_HE_O[6];
  ObjME a_ME_HF[6];
  ObjME a_ME_HB_lowRefPt[6];
  ObjME a_ME_HE_I_lowRefPt[6];
  ObjME a_ME_HE_O_lowRefPt[6];
  ObjME a_ME_HF_lowRefPt[6];
  ObjME a_ME_HB_mediumRefPt[6];
  ObjME a_ME_HE_I_mediumRefPt[6];
  ObjME a_ME_HE_O_mediumRefPt[6];
  ObjME a_ME_HF_mediumRefPt[6];
  ObjME a_ME_HB_highRefPt[6];
  ObjME a_ME_HE_I_highRefPt[6];
  ObjME a_ME_HE_O_highRefPt[6];
  ObjME a_ME_HF_highRefPt[6];

  ObjME mZMassME_;
  ObjME DPhiRefJetME_;

  std::vector<double> v_jetpt;
  std::vector<double> v_jeteta;
  std::vector<double> v_jetphi;

  std::vector<double> trigobj_pt;
  std::vector<double> trigobj_eta;
  std::vector<double> trigobj_phi;

  TLorentzVector muon_1;
  TLorentzVector muon_2;
  TLorentzVector Zhltreco;
  std::string fullpathName;
};

ZGammaplusJetsMonitor::ZGammaplusJetsMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      processName_(iConfig.getParameter<std::string>("processName")),
      triggerEventObject_(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("triggerEventObject"))),
      triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResultsLabel"))),
      pathName(iConfig.getParameter<std::string>("PathName")),
      moduleName(iConfig.getParameter<std::string>("ModuleName")),
      jetInputTag_(iConfig.getParameter<edm::InputTag>("jets")),
      jetToken_(mayConsume<reco::PFJetCollection>(jetInputTag_)),
      correctorToken_(mayConsume<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("corrector"))),
      muon_pt(iConfig.getParameter<double>("muonpt")),
      muon_eta(iConfig.getParameter<double>("muoneta")),
      pt_cut(iConfig.getParameter<double>("ptcut")),
      Z_DM(iConfig.getParameter<double>("Z_Dmass")),
      Z_Pt(iConfig.getParameter<double>("Z_pt")),
      dphi_cut(iConfig.getParameter<double>("DeltaPhi")),
      offline_cut(iConfig.getParameter<double>("OfflineCut")),
      isMuonPath_(iConfig.getParameter<bool>("isMuonPath")),
      directbalance_Binning(iConfig.getParameter<edm::ParameterSet>("histoPSet")
                                .getParameter<std::vector<double> >("directbalanceBinning")),
      TrObjPt_Binning(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("TrObjPtBinning")),
      jetpt_Binning(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("jetptBinning")) {}

ZGammaplusJetsMonitor::~ZGammaplusJetsMonitor() throw() {}

void ZGammaplusJetsMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& iSetup) {
  std::string histname, histtitle;
  std::string hist_obtag = "";
  std::string histtitle_obtag = "";
  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  if (isMuonPath_) {
    hist_obtag = "Z";
    histtitle_obtag = "Z ";
    histname = "DiMuonMass";
    histtitle = "DiMuonMass";
    bookME(ibooker, mZMassME_, histname, histtitle, 50, 71., 111., false);
  } else {
    hist_obtag = "Photon";
    histtitle_obtag = "Photon";
  }

  histname = "DPhi" + hist_obtag + "Jet";
  histtitle = "DPhi " + hist_obtag + " Jet";
  bookME(ibooker, DPhiRefJetME_, histname, histtitle, 100, 0., acos(-1.), false);

  bookMESub(ibooker, a_ME, sizeof(a_ME) / sizeof(a_ME[0]), hist_obtag, histtitle_obtag, "", "");
  bookMESub(ibooker, a_ME_HB, sizeof(a_ME_HB) / sizeof(a_ME_HB[0]), hist_obtag, histtitle_obtag, "HB", "(HB)", true);
  bookMESub(ibooker,
            a_ME_HE_I,
            sizeof(a_ME_HE_I) / sizeof(a_ME_HE_I[0]),
            hist_obtag,
            histtitle_obtag,
            "HEInner",
            "(HE Inner)",
            true);
  bookMESub(ibooker,
            a_ME_HE_O,
            sizeof(a_ME_HE_O) / sizeof(a_ME_HE_O[0]),
            hist_obtag,
            histtitle_obtag,
            "HEOuter",
            "(HE Outer)",
            true);
  bookMESub(ibooker, a_ME_HF, sizeof(a_ME_HF) / sizeof(a_ME_HF[0]), hist_obtag, histtitle_obtag, "HF", "(HF)", true);
  bookMESub(ibooker,
            a_ME_HB_lowRefPt,
            sizeof(a_ME_HB_lowRefPt) / sizeof(a_ME_HB_lowRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HB_lowRefPt",
            "(HB) lowRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HE_I_lowRefPt,
            sizeof(a_ME_HE_I_lowRefPt) / sizeof(a_ME_HE_I_lowRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HEInner_lowRefPt",
            "(HE Inner) lowRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HE_O_lowRefPt,
            sizeof(a_ME_HE_O_lowRefPt) / sizeof(a_ME_HE_O_lowRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HEOuter_lowRefPt",
            "(HE Outer) lowRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HF_lowRefPt,
            sizeof(a_ME_HF_lowRefPt) / sizeof(a_ME_HF_lowRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HF_lowRefPt",
            "(HF) lowRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HB_mediumRefPt,
            sizeof(a_ME_HB_mediumRefPt) / sizeof(a_ME_HB_mediumRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HB_mediumRefPt",
            "(HB) mediumRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HE_I_mediumRefPt,
            sizeof(a_ME_HE_I_mediumRefPt) / sizeof(a_ME_HE_I_mediumRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HEInner_mediumRefPt",
            "(HE Inner) mediumRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HE_O_mediumRefPt,
            sizeof(a_ME_HE_O_mediumRefPt) / sizeof(a_ME_HE_O_mediumRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HEOuter_mediumRefPt",
            "(HE Outer) mediumRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HF_mediumRefPt,
            sizeof(a_ME_HF_mediumRefPt) / sizeof(a_ME_HF_mediumRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HF_mediumRefPt",
            "(HF) mediumRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HB_highRefPt,
            sizeof(a_ME_HB_highRefPt) / sizeof(a_ME_HB_highRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HB_highRefPt",
            "(HB) highRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HE_I_highRefPt,
            sizeof(a_ME_HE_I_highRefPt) / sizeof(a_ME_HE_I_highRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HEInner_highRefPt",
            "(HE Inner) highRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HE_O_highRefPt,
            sizeof(a_ME_HE_O_highRefPt) / sizeof(a_ME_HE_O_highRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HEOuter_highRefPt",
            "(HE Outer) highRefPt",
            true);
  bookMESub(ibooker,
            a_ME_HF_highRefPt,
            sizeof(a_ME_HF_highRefPt) / sizeof(a_ME_HF_highRefPt[0]),
            hist_obtag,
            histtitle_obtag,
            "HF_highRefPt",
            "(HF) highRefPt",
            true);
}

void ZGammaplusJetsMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  v_jetpt.clear();
  v_jeteta.clear();
  v_jetphi.clear();

  trigobj_pt.clear();
  trigobj_eta.clear();
  trigobj_phi.clear();

  // ------------------ Get TriggerResults -----------------
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid())
    return;

  edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
  iEvent.getByToken(triggerEventObject_, aodTriggerEvent);
  if (!aodTriggerEvent.isValid())
    return;

  const edm::TriggerNames& triggerNames_ = iEvent.triggerNames(*triggerResults);  // all trigger names available

  bool passTrig = false;

  unsigned hltcfgIndex = hltConfig_.triggerIndex(fullpathName);
  unsigned index = triggerNames_.triggerIndex(fullpathName);  // find in which index is that path
  if (!(hltcfgIndex == index)) {
    edm::LogInfo("ZGammaplusJetsMonitor") << " Error in trigger index";
    return;
  }

  if (index > 0 && index < triggerNames_.size() &&
      triggerResults->accept(index)) {  //trigger is accepted and index is valid
    edm::LogInfo("ZGammaplusJetsMonitor") << " Trigger accepted";
    passTrig = true;
  }
  if (!passTrig) {
    edm::LogInfo("ZGammaplusJetsMonitor") << " Trigger did not pass";  // skip event if trigger fails
    return;
  }

  // ---------------- find module for this path ------------------------
  const unsigned int module_size = hltConfig_.size(index);
  std::vector<std::string> module_names = hltConfig_.moduleLabels(index);  // (pathname) works too
  if (!(module_size == module_names.size())) {
    edm::LogInfo("ZGammaplusJetsMonitor") << "ERROR IN MODULES COUNTING";
    return;
  }
  if (module_size == 0) {
    edm::LogInfo("ZGammaplusJetsMonitor") << "no modules in this path ?!?!";
    return;
  }

  // Find the module
  edm::InputTag moduleFilter;
  moduleFilter = edm::InputTag(moduleName, "", processName_);
  edm::LogInfo("ZGammaplusJetsMonitor") << " ModuleFilter " << moduleFilter;

  // check whether the module is packed up in TriggerEvent product
  trigger::size_type filterIndex_ = aodTriggerEvent->filterIndex(moduleFilter);
  edm::LogInfo("ZGammaplusJetsMonitor") << " filter index " << filterIndex_ << " | filter size  "
                                        << aodTriggerEvent->sizeFilters();
  if (filterIndex_ >= aodTriggerEvent->sizeFilters()) {
    return;
  }

  edm::LogInfo("ZGammaplusJetsMonitor") << " filter label|filter index" << moduleName << "|" << filterIndex_;

  const trigger::Vids& VIDS_ = aodTriggerEvent->filterIds(filterIndex_);
  const trigger::Keys& KEYS_ = aodTriggerEvent->filterKeys(filterIndex_);
  const trigger::size_type nI_ = VIDS_.size();
  const trigger::size_type nK_ = KEYS_.size();
  assert(nI_ == nK_);
  const trigger::TriggerObjectCollection& TOC(aodTriggerEvent->getObjects());
  for (trigger::size_type idx = 0; idx < nI_; ++idx) {
    const trigger::TriggerObject& TO(TOC[KEYS_[idx]]);
    //VIDS :  muons-->83 / jets-->85  / met-->87 / photon-->81
    edm::LogInfo("ZGammaplusJetsMonitor") << " idx:  " << idx << "| vid  " << VIDS_[idx] << "|"
                                          << " keys  " << KEYS_[idx] << "triggerobject: "
                                          << " obj_id " << TO.id() << " Pt " << TO.pt() << " eta " << TO.eta()
                                          << " phi " << TO.phi() << " mass " << TO.mass();
  }
  if (VIDS_[0] == 81) {  //photon
    for (const auto& key : KEYS_) {
      trigobj_pt.push_back(TOC[key].pt());
      trigobj_eta.push_back(TOC[key].eta());
      trigobj_phi.push_back(TOC[key].phi());
    }
  }
  if (VIDS_[0] == 83 && nK_ < 2) {  //muon
    edm::LogInfo("ZGammaplusJetsMonitor") << " under 2 objects cant have a dimuon";
    return;
  } else {
    for (const auto& key : KEYS_) {
      double pt = TOC[key].pt();
      double eta = TOC[key].eta();
      double phi = TOC[key].phi();
      double mass = TOC[key].mass();
      int id = TOC[key].id();
      unsigned int kCnt0 = 0;

      TLorentzVector v1;         //keep first muon
      if (std::abs(id) == 13) {  // check if it is a muon
        v1.SetPtEtaPhiM(pt, eta, phi, mass);
      } else {
        v1.SetPtEtaPhiM(0., 0., 0., 0.);
      }
      unsigned int kCnt1 = 0;
      for (const auto& key1 : KEYS_) {
        if (key != key1 && kCnt1 > kCnt0) {  // avoid double counting separate objs

          double pt2 = TOC[key1].pt();
          double eta2 = TOC[key1].eta();
          double phi2 = TOC[key1].phi();
          double mass2 = TOC[key1].mass();
          int id2 = TOC[key1].id();

          if ((id + id2) == 0) {  // check di-object system charge and flavor

            TLorentzVector v2;
            if (std::abs(id2) == 13) {  // check if it is a muon
              v2.SetPtEtaPhiM(pt2, eta2, phi2, mass2);
            } else {
              v2.SetPtEtaPhiM(0., 0., 0., 0.0);
            }

            muon_1 = v1;
            muon_2 = v2;
            bool muon_pass = muon_1.Pt() > muon_pt && muon_2.Pt() > muon_pt && std::abs(muon_1.Eta()) < muon_eta &&
                             std::abs(muon_2.Eta()) < muon_eta;
            if (!muon_pass) {
              return;
            }

            Zhltreco = muon_1 + muon_2;
            bool Z_pass = std::abs(Zhltreco.M() - 91.2) < Z_DM && Zhltreco.Pt() > Z_Pt;
            if (!Z_pass) {
              return;
            }
            trigobj_pt.push_back(Zhltreco.Pt());
            trigobj_eta.push_back(Zhltreco.Eta());
            trigobj_phi.push_back(Zhltreco.Phi());
          }  //end check di-object
          else {  //if not di-object
            return;
          }
        }  // end avoid duplicate objects
        kCnt1++;
      }  // key1
      kCnt0++;
    }  // key
  }  // end else

  // ---------------- module for Jet leg Jets --------------------------
  // index of last module executed in this Path
  const unsigned int moduleIndex = triggerResults->index(index);  // here would be HLTBool at the end
  edm::LogInfo("ZGammaplusJetsMonitor")
      << " Module Index " << moduleIndex - 1 << " Module Name  "
      << module_names[moduleIndex - 1];  // the second to last would be the last module that is saved
  assert(moduleIndex < module_size);

  // results from TriggerEvent product
  const std::string& ImoduleLabel = module_names[moduleIndex - 1];
  const std::string ImoduleType = hltConfig_.moduleType(ImoduleLabel);
  edm::LogInfo("ZGammaplusJetsMonitor") << ImoduleLabel << " |  " << ImoduleType;
  // check whether the module is packed up in TriggerEvent product
  const unsigned int filterIndex = aodTriggerEvent->filterIndex(edm::InputTag(ImoduleLabel, "", processName_));
  if (filterIndex >= aodTriggerEvent->sizeFilters()) {
    return;
  }
  const trigger::Vids& VIDS = aodTriggerEvent->filterIds(filterIndex);
  const trigger::Keys& KEYS = aodTriggerEvent->filterKeys(filterIndex);
  const trigger::size_type nI = VIDS.size();
  const trigger::size_type nK = KEYS.size();
  assert(nI == nK);
  const trigger::TriggerObjectCollection& objects(aodTriggerEvent->getObjects());
  for (trigger::size_type idx = 0; idx < nI; ++idx) {
    const trigger::TriggerObject& TO_(objects[KEYS[idx]]);
    //VIDS :  muons-->83 / jets-->85  / met-->87
    edm::LogInfo("ZGammaplusJetsMonitor")
        << " idx  " << idx << " vid  " << VIDS[idx] << "/"
        << " keys  " << KEYS[idx] << ": "
        << " obj_id " << TO_.id() << " " << TO_.pt() << " " << TO_.eta() << " " << TO_.phi() << " " << TO_.mass();
  }
  for (const auto& key : KEYS) {
    v_jetpt.push_back(objects[key].pt());
    v_jeteta.push_back(objects[key].eta());
    v_jetphi.push_back(objects[key].phi());
  }
  bool Jet_pass = (!v_jetpt.empty() && v_jetpt[0] >= pt_cut);
  if (!Jet_pass) {
    return;
  }
  double dphi = std::abs(v_jetphi[0] - trigobj_phi[0]);
  if (dphi > M_PI) {
    dphi = 2 * M_PI - dphi;
  }
  if (dphi < dphi_cut) {  // be sure is back to back
    return;
  }
  // --- offline Jets -----
  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  edm::Handle<reco::JetCorrector> Corrector;
  iEvent.getByToken(correctorToken_, Corrector);
  double leading_JetPt = -10.0;
  double leading_JetEta = -10.0;
  double leading_JetPhi = -10.0;
  int ind = 0;

  if (!jetHandle.isValid()) {
    edm::LogWarning("ZGammaplusJetsMonitor")
        << "skipping events because the collection " << jetInputTag_.label().c_str() << " is not available";
    return;
  }
  for (auto const& j : *jetHandle) {
    if (Corrector.isValid()) {
      double jec = Corrector->correction(j);
      double cor_jet = jec * j.pt();
      if (cor_jet > leading_JetPt) {
        leading_JetPt = cor_jet;
        leading_JetEta = j.eta();
        leading_JetPhi = j.phi();
      }
    } else if (!Corrector.isValid() && ind == 0) {
      leading_JetPt = j.pt();
      leading_JetEta = j.eta();
      leading_JetPhi = j.phi();
      ind += 1;
    }
  }
  if (leading_JetPt < offline_cut) {  // offline cuts
    return;
  }

  if (!(isMatched(v_jeteta[0], v_jetphi[0], leading_JetEta, leading_JetPhi))) {
    return;
  }

  double DirectBalance_ = v_jetpt[0] / trigobj_pt[0];
  double DifJ1PtTrObjPt_ = v_jetpt[0] - trigobj_pt[0];
  double asymmetry = (trigobj_pt[0] - v_jetpt[0]) / (trigobj_pt[0] + v_jetpt[0]);

  // ------------------------- Filling Histos -----------------------------------
  if (isMuonPath_) {
    mZMassME_.numerator->Fill(Zhltreco.M());
  }
  DPhiRefJetME_.numerator->Fill(dphi);
  fillME(a_ME, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
  if (isBarrel(v_jeteta[0])) {
    fillME(a_ME_HB, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    if (islowRefPt(trigobj_pt[0])) {
      fillME(a_ME_HB_lowRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ismediumRefPt(trigobj_pt[0])) {
      fillME(a_ME_HB_mediumRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ishighRefPt(trigobj_pt[0])) {
      fillME(a_ME_HB_highRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    }
  }  // end is barrel
  if (isEndCapInner(v_jeteta[0])) {
    fillME(a_ME_HE_I, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    if (islowRefPt(trigobj_pt[0])) {
      fillME(a_ME_HE_I_lowRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ismediumRefPt(trigobj_pt[0])) {
      fillME(a_ME_HE_I_mediumRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ishighRefPt(trigobj_pt[0])) {
      fillME(a_ME_HE_I_highRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    }
  }  // end is endcap-inner
  if (isEndCapOuter(v_jeteta[0])) {
    fillME(a_ME_HE_O, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    if (islowRefPt(trigobj_pt[0])) {
      fillME(a_ME_HE_O_lowRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ismediumRefPt(trigobj_pt[0])) {
      fillME(a_ME_HE_O_mediumRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ishighRefPt(trigobj_pt[0])) {
      fillME(a_ME_HE_O_highRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    }
  }  // end is endcap-outer
  if (isForward(v_jeteta[0])) {
    fillME(a_ME_HF, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    if (islowRefPt(trigobj_pt[0])) {
      fillME(a_ME_HF_lowRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ismediumRefPt(trigobj_pt[0])) {
      fillME(a_ME_HF_mediumRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    } else if (ishighRefPt(trigobj_pt[0])) {
      fillME(a_ME_HF_highRefPt, DirectBalance_, DifJ1PtTrObjPt_, asymmetry, trigobj_pt[0], v_jetpt[0], true);
    }
  }  // end is Forward
}
// ----- Reference'Pt categories: low[30,50), medium[50,100), high[100,inf) -----
bool ZGammaplusJetsMonitor::islowRefPt(double refPt) {
  bool output = false;
  if (refPt >= 30. && refPt < 50.)
    output = true;
  return output;
}

bool ZGammaplusJetsMonitor::ismediumRefPt(double refPt) {
  bool output = false;
  if (refPt >= 50. && refPt < 100.)
    output = true;
  return output;
}

bool ZGammaplusJetsMonitor::ishighRefPt(double refPt) {
  bool output = false;
  if (refPt >= 100.)
    output = true;
  return output;
}
// --------- detector areas ---------------------------------------------------
bool ZGammaplusJetsMonitor::isBarrel(double eta) {
  bool output = false;
  if (std::abs(eta) <= 1.3)
    output = true;
  return output;
}

bool ZGammaplusJetsMonitor::isEndCapInner(double eta) {
  bool output = false;
  if (std::abs(eta) <= 2.5 && std::abs(eta) > 1.3)
    output = true;
  return output;
}

bool ZGammaplusJetsMonitor::isEndCapOuter(double eta) {
  bool output = false;
  if (std::abs(eta) <= 3.0 && std::abs(eta) > 2.5)
    output = true;
  return output;
}

bool ZGammaplusJetsMonitor::isForward(double eta) {
  bool output = false;
  if (std::abs(eta) > 3.0)
    output = true;
  return output;
}
// --------- Matching ---------------------------------------------------------------
bool ZGammaplusJetsMonitor::isMatched(double hltJetEta, double hltJetPhi, double OffJetEta, double OffJetPhi) {
  bool output = false;
  double DRMatched2 = 0.16;
  double dR2 = deltaR2(hltJetEta, hltJetPhi, OffJetEta, OffJetPhi);
  if (dR2 < DRMatched2)
    output = true;
  return output;
}

void ZGammaplusJetsMonitor::fillME(ObjME* a_me,
                                   const double directbalance_,
                                   const double Difjetref_,
                                   const double Asymmetry_,
                                   const double ReferencePt_,
                                   const double Jetpt_,
                                   const bool doDirectBalancevsReferencePt) {
  a_me[0].numerator->Fill(directbalance_);  // index 0 = DirectBalance
  a_me[1].numerator->Fill(Difjetref_);      // index 1 = Leading JetPt minus Reference Pt
  a_me[2].numerator->Fill(Asymmetry_);      // index 2 =  asymmetry
  a_me[3].numerator->Fill(ReferencePt_);    // index 3 = Reference Pt
  a_me[4].numerator->Fill(Jetpt_);          // index 4 = Jet Pt
  if (doDirectBalancevsReferencePt) {
    a_me[5].numerator->Fill(ReferencePt_, directbalance_);  // index 5 = Balance vs Reference' Pt
  }
}

void ZGammaplusJetsMonitor::bookMESub(DQMStore::IBooker& Ibooker,
                                      ObjME* a_me,
                                      const int len_,
                                      const std::string& h_Name,
                                      const std::string& h_Title,
                                      const std::string& h_subOptName,
                                      const std::string& hSubT,
                                      const bool doDirectBalancevsReferencePt,
                                      const bool bookDen) {
  std::string hName = h_Name;
  std::string hTitle = h_Title;
  const std::string hSubN = h_subOptName.empty() ? "" : "_" + h_subOptName;

  int nbin_DifJetRef = DifJetRefPT_Binning.nbins;
  double maxbin_DifJetRef = DifJetRefPT_Binning.xmax;
  double minbin_DifJetRef = DifJetRefPT_Binning.xmin;

  hName = "DirectBalance" + hSubN;
  hTitle = " DirectBalance " + hSubT;
  bookME(Ibooker, a_me[0], hName, hTitle, directbalance_Binning, bookDen);
  setMETitle(a_me[0], "HLTJetPt/" + h_Name + "Pt", "events");

  hName = "JetPt1_minus_" + h_Name + "Pt" + hSubN;
  hTitle = "LeadingJet Pt minus " + h_Name + " Pt " + hSubT;
  bookME(Ibooker, a_me[1], hName, hTitle, nbin_DifJetRef, minbin_DifJetRef, maxbin_DifJetRef, bookDen);
  setMETitle(a_me[1], "Pt dif [GeV]", "events");

  hName = h_Name + "JetAsymmetry" + hSubN;
  hTitle = h_Title + " Jet Asymmetry " + hSubT;
  bookME(Ibooker, a_me[2], hName, hTitle, directbalance_Binning, bookDen);
  setMETitle(a_me[2], hTitle, "events");

  hName = h_Name + "pT" + hSubN;
  hTitle = h_Title + " pT " + hSubT;
  bookME(Ibooker, a_me[3], hName, hTitle, TrObjPt_Binning, bookDen);
  setMETitle(a_me[3], h_Title + " pT [GeV]", "events / [GeV]");

  hName = "JetpT" + hSubN;
  hTitle = "Jet pT " + hSubN;
  bookME(Ibooker, a_me[4], hName, hTitle, jetpt_Binning, bookDen);
  setMETitle(a_me[4], hTitle + " [GeV]", "events / [GeV]");

  if (doDirectBalancevsReferencePt) {
    hName = "DirectBalanceVs" + h_Name + "Pt" + hSubN;
    hTitle = "Direct Balance vs " + h_Title + " Pt " + hSubT;
    bookME(Ibooker, a_me[5], hName, hTitle, TrObjPt_Binning, directbalance_Binning, bookDen);
    setMETitle(a_me[5], h_Title + " pt", "Direct Balance");
  }
}

void ZGammaplusJetsMonitor::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  TPRegexp pattern(pathName);
  // https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideHighLevelTrigger#Access_to_the_HLT_configuration
  // "init" return value indicates whether intitialisation has succeeded
  // "changed" parameter indicates whether the config has actually changed
  bool changed(true);
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      std::vector<std::string> triggerPaths = hltConfig_.triggerNames();
      for (const auto& PATHNAME : triggerPaths) {
        edm::LogInfo("ZGammaplusJetsMonitor::dqmBeginRun ") << PATHNAME;
        if (TString(PATHNAME).Contains(pattern)) {
          fullpathName = PATHNAME;
        }
      }
    }
  } else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    edm::LogError("ZGammaplusJetsMonitor") << " HLT config extraction failure with process name " << processName_;
    // In this case, all access methods will return empty values!
  }
}

void ZGammaplusJetsMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/JME/ZGammaPlusJets");
  desc.add<std::string>("processName", "HLT");
  desc.add<edm::InputTag>("triggerEventObject", edm::InputTag("hltTriggerSummaryAOD::HLT"));
  desc.add<edm::InputTag>("TriggerResultsLabel", edm::InputTag("TriggerResults::HLT"));
  desc.add<std::string>("PathName", "");
  desc.add<std::string>("ModuleName", "");
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsPuppi"));
  desc.add<edm::InputTag>("corrector", edm::InputTag("ak4PFPuppiL1FastL2L3Corrector"));

  desc.add<double>("muonpt", 20.);
  desc.add<double>("muoneta", 2.3);
  desc.add<double>("ptcut", 30.);
  desc.add<double>("Z_Dmass", 20.);
  desc.add<double>("Z_pt", 30.);
  desc.add<double>("DeltaPhi", 2.7);
  desc.add<double>("OfflineCut", 20.0);
  desc.add<bool>("isMuonPath", true);

  edm::ParameterSetDescription histoPSet;

  std::vector<double> bins = {
      -3.99, -3.97, -3.95, -3.93, -3.91, -3.89, -3.87, -3.85, -3.83, -3.81, -3.79, -3.77, -3.75, -3.73, -3.71, -3.69,
      -3.67, -3.65, -3.63, -3.61, -3.59, -3.57, -3.55, -3.53, -3.51, -3.49, -3.47, -3.45, -3.43, -3.41, -3.39, -3.37,
      -3.35, -3.33, -3.31, -3.29, -3.27, -3.25, -3.23, -3.21, -3.19, -3.17, -3.15, -3.13, -3.11, -3.09, -3.07, -3.05,
      -3.03, -3.01, -2.99, -2.97, -2.95, -2.93, -2.91, -2.89, -2.87, -2.85, -2.83, -2.81, -2.79, -2.77, -2.75, -2.73,
      -2.71, -2.69, -2.67, -2.65, -2.63, -2.61, -2.59, -2.57, -2.55, -2.53, -2.51, -2.49, -2.47, -2.45, -2.43, -2.41,
      -2.39, -2.37, -2.35, -2.33, -2.31, -2.29, -2.27, -2.25, -2.23, -2.21, -2.19, -2.17, -2.15, -2.13, -2.11, -2.09,
      -2.07, -2.05, -2.03, -2.01, -1.99, -1.97, -1.95, -1.93, -1.91, -1.89, -1.87, -1.85, -1.83, -1.81, -1.79, -1.77,
      -1.75, -1.73, -1.71, -1.69, -1.67, -1.65, -1.63, -1.61, -1.59, -1.57, -1.55, -1.53, -1.51, -1.49, -1.47, -1.45,
      -1.43, -1.41, -1.39, -1.37, -1.35, -1.33, -1.31, -1.29, -1.27, -1.25, -1.23, -1.21, -1.19, -1.17, -1.15, -1.13,
      -1.11, -1.09, -1.07, -1.05, -1.03, -1.01, -0.99, -0.97, -0.95, -0.93, -0.91, -0.89, -0.87, -0.85, -0.83, -0.81,
      -0.79, -0.77, -0.75, -0.73, -0.71, -0.69, -0.67, -0.65, -0.63, -0.61, -0.59, -0.57, -0.55, -0.53, -0.51, -0.49,
      -0.47, -0.45, -0.43, -0.41, -0.39, -0.37, -0.35, -0.33, -0.31, -0.29, -0.27, -0.25, -0.23, -0.21, -0.19, -0.17,
      -0.15, -0.13, -0.11, -0.09, -0.07, -0.05, -0.03, -0.01, 0.01,  0.03,  0.05,  0.07,  0.09,  0.11,  0.13,  0.15,
      0.17,  0.19,  0.21,  0.23,  0.25,  0.27,  0.29,  0.31,  0.33,  0.35,  0.37,  0.39,  0.41,  0.43,  0.45,  0.47,
      0.49,  0.51,  0.53,  0.55,  0.57,  0.59,  0.61,  0.63,  0.65,  0.67,  0.69,  0.71,  0.73,  0.75,  0.77,  0.79,
      0.81,  0.83,  0.85,  0.87,  0.89,  0.91,  0.93,  0.95,  0.97,  0.99,  1.01,  1.03,  1.05,  1.07,  1.09,  1.11,
      1.13,  1.15,  1.17,  1.19,  1.21,  1.23,  1.25,  1.27,  1.29,  1.31,  1.33,  1.35,  1.37,  1.39,  1.41,  1.43,
      1.45,  1.47,  1.49,  1.51,  1.53,  1.55,  1.57,  1.59,  1.61,  1.63,  1.65,  1.67,  1.69,  1.71,  1.73,  1.75,
      1.77,  1.79,  1.81,  1.83,  1.85,  1.87,  1.89,  1.91,  1.93,  1.95,  1.97,  1.99,  2.01,  2.03,  2.05,  2.07,
      2.09,  2.11,  2.13,  2.15,  2.17,  2.19,  2.21,  2.23,  2.25,  2.27,  2.29,  2.31,  2.33,  2.35,  2.37,  2.39,
      2.41,  2.43,  2.45,  2.47,  2.49,  2.51,  2.53,  2.55,  2.57,  2.59,  2.61,  2.63,  2.65,  2.67,  2.69,  2.71,
      2.73,  2.75,  2.77,  2.79,  2.81,  2.83,  2.85,  2.87,  2.89,  2.91,  2.93,  2.95,  2.97,  2.99,  3.01,  3.03,
      3.05,  3.07,  3.09,  3.11,  3.13,  3.15,  3.17,  3.19,  3.21,  3.23,  3.25,  3.27,  3.29,  3.31,  3.33,  3.35,
      3.37,  3.39,  3.41,  3.43,  3.45,  3.47,  3.49,  3.51,  3.53,  3.55,  3.57,  3.59,  3.61,  3.63,  3.65,  3.67,
      3.69,  3.71,  3.73,  3.75,  3.77,  3.79,  3.81,  3.83,  3.85,  3.87,  3.89,  3.91,  3.93,  3.95,  3.97,  3.99,
      4.01,  4.03,  4.05,  4.07,  4.09,  4.11,  4.13,  4.15,  4.17,  4.19,  4.21,  4.23,  4.25,  4.27,  4.29,  4.31,
      4.33,  4.35,  4.37,  4.39,  4.41,  4.43,  4.45,  4.47,  4.49,  4.51,  4.53,  4.55,  4.57,  4.59,  4.61,  4.63,
      4.65,  4.67,  4.69,  4.71,  4.73,  4.75,  4.77,  4.79,  4.81,  4.83,  4.85,  4.87,  4.89,  4.91,  4.93,  4.95,
      4.97,  4.99,  5.01,  5.03,  5.05,  5.07,  5.09,  5.11,  5.13,  5.15,  5.17,  5.19,  5.21,  5.23,  5.25,  5.27,
      5.29,  5.31,  5.33,  5.35,  5.37,  5.39,  5.41,  5.43,  5.45,  5.47,  5.49,  5.51,  5.53,  5.55,  5.57,  5.59,
      5.61,  5.63,  5.65,  5.67,  5.69,  5.71,  5.73,  5.75,  5.77,  5.79,  5.81,  5.83,  5.85,  5.87,  5.89,  5.91,
      5.93,  5.95,  5.97,  5.99};

  histoPSet.add<std::vector<double> >("directbalanceBinning", bins);

  std::vector<double> bins_ = {12, 15,  20,  25,  30,  35,  40,  45,  50,  60,   70,
                               85, 105, 130, 175, 230, 300, 400, 500, 700, 1000, 1500};  // Z or photon pT Binning
  histoPSet.add<std::vector<double> >("TrObjPtBinning", bins_);
  std::vector<double> Jbins_ = {
      0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
      170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};  // Jet pT Binning
  histoPSet.add<std::vector<double> >("jetptBinning", Jbins_);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("zgammajetsmonitoring", desc);
}

DEFINE_FWK_MODULE(ZGammaplusJetsMonitor);

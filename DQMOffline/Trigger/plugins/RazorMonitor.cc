// -----------------------------
//
// Offline DQM for razor triggers. The razor inclusive analysis measures trigger efficiency
// in SingleElectron events (orthogonal to analysis), as a 2D function of the razor variables
// M_R and R^2. Also monitor dPhi_R, used offline for  QCD and/or detector-related MET tail
// rejection.
// Based on DQMOffline/Trigger/plugins/METMonitor.cc
//
// -----------------------------
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "HLTrigger/JetMET/interface/HLTRHemisphere.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <TVector3.h>

class RazorMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  RazorMonitor(const edm::ParameterSet&);
  ~RazorMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static double CalcMR(const math::XYZTLorentzVector& ja, const math::XYZTLorentzVector& jb);
  static double CalcR(double MR,
                      const math::XYZTLorentzVector& ja,
                      const math::XYZTLorentzVector& jb,
                      const edm::Handle<std::vector<reco::PFMET> >& met);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector> > theHemispheres_;

  std::vector<double> rsq_binning_;
  std::vector<double> mr_binning_;
  std::vector<double> dphiR_binning_;

  ObjME MR_ME_;
  ObjME Rsq_ME_;
  ObjME dPhiR_ME_;
  ObjME MRVsRsq_ME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  unsigned int njets_;
  float rsqCut_;
  float mrCut_;
};

// -----------------------------
//
// Offline DQM for razor triggers. The razor inclusive analysis measures trigger efficiency
// in SingleElectron events (orthogonal to analysis), as a 2D function of the razor variables
// M_R and R^2. Also monitor dPhi_R, used offline for  QCD and/or detector-related MET tail
// rejection.
// Based on DQMOffline/Trigger/plugins/METMonitor.*
//
// -----------------------------

RazorMonitor::RazorMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      metToken_(consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("met"))),
      jetToken_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      theHemispheres_(
          consumes<std::vector<math::XYZTLorentzVector> >(iConfig.getParameter<edm::InputTag>("hemispheres"))),
      rsq_binning_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("rsqBins")),
      mr_binning_(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("mrBins")),
      dphiR_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("dphiRBins")),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      njets_(iConfig.getParameter<unsigned int>("njets")),
      rsqCut_(iConfig.getParameter<double>("rsqCut")),
      mrCut_(iConfig.getParameter<double>("mrCut")) {}

RazorMonitor::~RazorMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void RazorMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on())
    num_genTriggerEventFlag_->initRun(iRun, iSetup);
  if (den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on())
    den_genTriggerEventFlag_->initRun(iRun, iSetup);

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

  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  // 1D hist, MR
  histname = "MR";
  histtitle = "PF MR";
  bookME(ibooker, MR_ME_, histname, histtitle, mr_binning_);
  setMETitle(MR_ME_, "PF M_{R} [GeV]", "events / [GeV]");

  // 1D hist, Rsq
  histname = "Rsq";
  histtitle = "PF Rsq";
  bookME(ibooker, Rsq_ME_, histname, histtitle, rsq_binning_);
  setMETitle(Rsq_ME_, "PF R^{2}", "events");

  // 1D hist, dPhiR
  histname = "dPhiR";
  histtitle = "dPhiR";
  bookME(ibooker, dPhiR_ME_, histname, histtitle, dphiR_binning_);
  setMETitle(dPhiR_ME_, "dPhi_{R}", "events");

  // 2D hist, MR & Rsq
  histname = "MRVsRsq";
  histtitle = "PF MR vs PF Rsq";
  bookME(ibooker, MRVsRsq_ME_, histname, histtitle, mr_binning_, rsq_binning_);
  setMETitle(MRVsRsq_ME_, "M_{R} [GeV]", "R^{2}");
}

void RazorMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  //met collection
  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  reco::PFMET pfmet = metHandle->front();
  if (!metSelection_(pfmet))
    return;

  //jet collection, track # of jets for two working points
  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  std::vector<reco::PFJet> jets;
  if (jetHandle->size() < njets_)
    return;
  for (auto const& j : *jetHandle) {
    if (jetSelection_(j))
      jets.push_back(j);
  }
  if (jets.size() < njets_)
    return;

  //razor hemisphere clustering from previous step
  edm::Handle<vector<math::XYZTLorentzVector> > hemispheres;
  iEvent.getByToken(theHemispheres_, hemispheres);

  if (not hemispheres.isValid()) {
    return;
  }

  if (hemispheres
          ->empty()) {  // the Hemisphere Maker will produce an empty collection of hemispheres if # of jets is too high
    edm::LogError("DQM_HLT_Razor") << "Cannot calculate M_R and R^2 because there are too many jets! (trigger passed "
                                      "automatically without forming the hemispheres)"
                                   << endl;
    return;
  }

  // should always have 2 hemispheres -- no muons included (c. 2017), if not return invalid hemisphere collection
  // retaining check for hemisphere size 5 or 10 which correspond to the one or two muon case for possible future use
  if (!hemispheres->empty() && hemispheres->size() != 2 && hemispheres->size() != 5 && hemispheres->size() != 10) {
    edm::LogError("DQM_HLT_Razor") << "Invalid hemisphere collection!  hemispheres->size() = " << hemispheres->size()
                                   << endl;
    return;
  }

  //calculate razor variables, with hemispheres pT-ordered
  double MR = 0, R = 0;
  if (hemispheres->at(1).Pt() > hemispheres->at(0).Pt()) {
    MR = CalcMR(hemispheres->at(1), hemispheres->at(0));
    R = CalcR(MR, hemispheres->at(1), hemispheres->at(0), metHandle);
  } else {
    MR = CalcMR(hemispheres->at(0), hemispheres->at(1));
    R = CalcR(MR, hemispheres->at(0), hemispheres->at(1), metHandle);
  }

  double Rsq = R * R;
  double dPhiR = abs(deltaPhi(hemispheres->at(0).Phi(), hemispheres->at(1).Phi()));

  //apply offline selection cuts
  if (Rsq < rsqCut_ && MR < mrCut_)
    return;

  // applying selection for denominator
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  // filling histograms (denominator)
  if (Rsq >= rsqCut_) {
    MR_ME_.denominator->Fill(MR);
  }

  if (MR >= mrCut_) {
    Rsq_ME_.denominator->Fill(Rsq);
  }

  dPhiR_ME_.denominator->Fill(dPhiR);

  MRVsRsq_ME_.denominator->Fill(MR, Rsq);

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  // filling histograms (numerator)
  if (Rsq >= rsqCut_) {
    MR_ME_.numerator->Fill(MR);
  }

  if (MR >= mrCut_) {
    Rsq_ME_.numerator->Fill(Rsq);
  }

  dPhiR_ME_.numerator->Fill(dPhiR);

  MRVsRsq_ME_.numerator->Fill(MR, Rsq);
}

void RazorMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/SUSY/Razor");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("hemispheres", edm::InputTag("hemispheresDQM"))
      ->setComment("hemisphere jets used to compute razor variables");
  desc.add<std::string>("metSelection", "pt > 0");

  // from 2016 offline selection
  desc.add<std::string>("jetSelection", "pt > 80");
  desc.add<unsigned int>("njets", 2);
  desc.add<double>("mrCut", 300);
  desc.add<double>("rsqCut", 0.15);

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

  //binning from 2016 offline selection
  edm::ParameterSetDescription histoPSet;
  std::vector<double> mrbins = {0., 100., 200., 300., 400., 500., 575., 650., 750., 900., 1200., 1600., 2500., 4000.};
  histoPSet.add<std::vector<double> >("mrBins", mrbins);

  std::vector<double> rsqbins = {0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.41, 0.52, 0.64, 0.8, 1.5};
  histoPSet.add<std::vector<double> >("rsqBins", rsqbins);

  std::vector<double> dphirbins = {0., 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.2};
  histoPSet.add<std::vector<double> >("dphiRBins", dphirbins);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("razorMonitoring", desc);
}

//CalcMR and CalcR borrowed from HLTRFilter.cc
double RazorMonitor::CalcMR(const math::XYZTLorentzVector& ja, const math::XYZTLorentzVector& jb) {
  if (ja.Pt() <= 0.1)
    return -1;

  double A = ja.P();
  double B = jb.P();
  double az = ja.Pz();
  double bz = jb.Pz();
  TVector3 jaT, jbT;
  jaT.SetXYZ(ja.Px(), ja.Py(), 0.0);
  jbT.SetXYZ(jb.Px(), jb.Py(), 0.0);
  double ATBT = (jaT + jbT).Mag2();

  double MR = sqrt((A + B) * (A + B) - (az + bz) * (az + bz) -
                   (jbT.Dot(jbT) - jaT.Dot(jaT)) * (jbT.Dot(jbT) - jaT.Dot(jaT)) / (jaT + jbT).Mag2());
  double mybeta = (jbT.Dot(jbT) - jaT.Dot(jaT)) / sqrt(ATBT * ((A + B) * (A + B) - (az + bz) * (az + bz)));

  double mygamma = 1. / sqrt(1. - mybeta * mybeta);

  //use gamma times MRstar
  return MR * mygamma;
}

double RazorMonitor::CalcR(double MR,
                           const math::XYZTLorentzVector& ja,
                           const math::XYZTLorentzVector& jb,
                           const edm::Handle<std::vector<reco::PFMET> >& inputMet) {
  //now we can calculate MTR
  const math::XYZVector met = (inputMet->front()).momentum();

  double MTR = sqrt(0.5 * (met.R() * (ja.Pt() + jb.Pt()) - met.Dot(ja.Vect() + jb.Vect())));

  //filter events
  return float(MTR) / float(MR);  //R
}

// Define this as a plug-in
DEFINE_FWK_MODULE(RazorMonitor);

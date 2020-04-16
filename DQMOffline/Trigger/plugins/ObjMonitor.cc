#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DQMOffline/Trigger/plugins/METDQM.h"
#include "DQMOffline/Trigger/plugins/JetDQM.h"
#include "DQMOffline/Trigger/plugins/HTDQM.h"
#include "DQMOffline/Trigger/plugins/HMesonGammaDQM.h"

class ObjMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  ObjMonitor(const edm::ParameterSet&);
  ~ObjMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  bool looseJetId(const double& abseta,
                  const double& NHF,
                  const double& NEMF,
                  const double& CHF,
                  const double& CEMF,
                  const unsigned& NumNeutralParticles,
                  const unsigned& CHM);

  bool tightJetId(const double& abseta,
                  const double& NHF,
                  const double& NEMF,
                  const double& CHF,
                  const double& CEMF,
                  const unsigned& NumNeutralParticles,
                  const unsigned& CHM);

  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phoToken_;
  edm::EDGetTokenT<reco::TrackCollection> trkToken_;

  //objects to plot
  //add your own with corresponding switch
  bool do_met_;
  METDQM metDQM_;
  bool do_jet_;
  JetDQM jetDQM_;
  bool do_ht_;
  HTDQM htDQM_;
  bool do_hmg_;
  HMesonGammaDQM hmgDQM_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  std::string jetId_;
  StringCutObjectSelector<reco::PFJet, true> htjetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Photon, true> phoSelection_;
  StringCutObjectSelector<reco::Track, true> trkSelection_;

  unsigned njets_;
  unsigned nelectrons_;
  unsigned nmuons_;
  unsigned nphotons_;
  unsigned nmesons_;
};

ObjMonitor::ObjMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      metToken_(consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("met"))),
      jetToken_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      eleToken_(mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"))),
      muoToken_(mayConsume<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      phoToken_(mayConsume<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("photons"))),
      trkToken_(mayConsume<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      do_met_(iConfig.getParameter<bool>("doMETHistos")),
      do_jet_(iConfig.getParameter<bool>("doJetHistos")),
      do_ht_(iConfig.getParameter<bool>("doHTHistos")),
      do_hmg_(iConfig.getParameter<bool>("doHMesonGammaHistos")),
      num_genTriggerEventFlag_(std::make_unique<GenericTriggerEventFlag>(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(std::make_unique<GenericTriggerEventFlag>(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      jetId_(iConfig.getParameter<std::string>("jetId")),
      htjetSelection_(iConfig.getParameter<std::string>("htjetSelection")),
      eleSelection_(iConfig.getParameter<std::string>("eleSelection")),
      muoSelection_(iConfig.getParameter<std::string>("muoSelection")),
      phoSelection_(iConfig.getParameter<std::string>("phoSelection")),
      trkSelection_(iConfig.getParameter<std::string>("trkSelection")),
      njets_(iConfig.getParameter<int>("njets")),
      nelectrons_(iConfig.getParameter<int>("nelectrons")),
      nmuons_(iConfig.getParameter<int>("nmuons")),
      nphotons_(iConfig.getParameter<int>("nphotons")),
      nmesons_(iConfig.getParameter<int>("nmesons")) {
  if (do_met_) {
    metDQM_.initialise(iConfig);
  }
  if (do_jet_) {
    jetDQM_.initialise(iConfig);
  }
  if (do_ht_) {
    htDQM_.initialise(iConfig);
  }
  if (do_hmg_) {
    hmgDQM_.initialise(iConfig);
  }
}

ObjMonitor::~ObjMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void ObjMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
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

  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  if (do_met_)
    metDQM_.bookHistograms(ibooker);
  if (do_jet_)
    jetDQM_.bookHistograms(ibooker);
  if (do_ht_)
    htDQM_.bookHistograms(ibooker);
  if (do_hmg_)
    hmgDQM_.bookHistograms(ibooker);
}

void ObjMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  reco::PFMET pfmet = metHandle->front();
  if (!metSelection_(pfmet))
    return;

  float met = pfmet.pt();
  float phi = pfmet.phi();

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  std::vector<reco::PFJet> jets;
  std::vector<reco::PFJet> htjets;
  if (jetHandle->size() < njets_)
    return;
  for (auto const& j : *jetHandle) {
    if (jetSelection_(j)) {
      if (jetId_ == "loose" || jetId_ == "tight") {
        double abseta = abs(j.eta());
        double NHF = j.neutralHadronEnergyFraction();
        double NEMF = j.neutralEmEnergyFraction();
        double CHF = j.chargedHadronEnergyFraction();
        double CEMF = j.chargedEmEnergyFraction();
        unsigned NumNeutralParticles = j.neutralMultiplicity();
        unsigned CHM = j.chargedMultiplicity();
        bool passId = (jetId_ == "loose" && looseJetId(abseta, NHF, NEMF, CHF, CEMF, NumNeutralParticles, CHM)) ||
                      (jetId_ == "tight" && tightJetId(abseta, NHF, NEMF, CHF, CEMF, NumNeutralParticles, CHM));
        if (passId)
          jets.push_back(j);
      } else
        jets.push_back(j);
    }
    if (htjetSelection_(j))
      htjets.push_back(j);
  }
  if (jets.size() < njets_)
    return;

  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken(eleToken_, eleHandle);
  std::vector<reco::GsfElectron> electrons;
  if (eleHandle->size() < nelectrons_)
    return;
  for (auto const& e : *eleHandle) {
    if (eleSelection_(e))
      electrons.push_back(e);
  }
  if (electrons.size() < nelectrons_)
    return;

  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken(muoToken_, muoHandle);
  if (muoHandle->size() < nmuons_)
    return;
  std::vector<reco::Muon> muons;
  for (auto const& m : *muoHandle) {
    if (muoSelection_(m))
      muons.push_back(m);
  }
  if (muons.size() < nmuons_)
    return;

  edm::Handle<reco::PhotonCollection> phoHandle;
  iEvent.getByToken(phoToken_, phoHandle);
  if (phoHandle->size() < nphotons_)
    return;
  std::vector<reco::Photon> photons;
  for (auto const& m : *phoHandle) {
    if (phoSelection_(m))
      photons.push_back(m);
  }
  if (photons.size() < nphotons_)
    return;

  std::vector<TLorentzVector> passedMesons;
  if (do_hmg_) {
    edm::Handle<reco::TrackCollection> trkHandle;
    iEvent.getByToken(trkToken_, trkHandle);
    // find isolated mesons (phi or rho)
    TLorentzVector t1, t2;
    float hadronMassHyp[2] = {0.1396, 0.4937};  // pi or K mass
    float loMassLim[2] = {0.5, 0.9};            // rho or phi mass
    float hiMassLim[2] = {1.0, 1.11};           // rho or phi mass

    for (size_t i = 0; i < trkHandle->size(); ++i) {
      const reco::Track trk1 = trkHandle->at(i);
      if (!trkSelection_(trk1))
        continue;
      for (size_t j = i + 1; j < trkHandle->size(); ++j) {
        const reco::Track trk2 = trkHandle->at(j);
        if (!trkSelection_(trk2))
          continue;
        if (trk1.charge() * trk2.charge() != -1)
          continue;

        for (unsigned hyp = 0; hyp < 2; ++hyp) {
          t1.SetPtEtaPhiM(trk1.pt(), trk1.eta(), trk1.phi(), hadronMassHyp[hyp]);
          t2.SetPtEtaPhiM(trk2.pt(), trk2.eta(), trk2.phi(), hadronMassHyp[hyp]);
          TLorentzVector mesCand = t1 + t2;

          // cuts
          if (mesCand.M() < loMassLim[hyp] || mesCand.M() > hiMassLim[hyp])
            continue;  //mass
          if (mesCand.Pt() < 35. || fabs(mesCand.Rapidity()) > 2.1)
            continue;  //pT eta

          // isolation
          float absIso = 0.;
          for (size_t k = 0; k < trkHandle->size(); ++k) {
            if (k == i || k == j)
              continue;
            const reco::Track trkN = trkHandle->at(k);
            if (trkN.charge() == 0 || trkN.pt() < 0.5 || (trkN.dz() > 0.1) ||
                deltaR(trkN.eta(), trkN.phi(), mesCand.Eta(), mesCand.Phi()) > 0.5)
              continue;
            absIso += trkN.pt();
          }
          if (absIso / mesCand.Pt() > 0.2)
            continue;
          passedMesons.push_back(mesCand);
        }
      }
    }
    if (passedMesons.size() < nmesons_)
      return;
  }

  bool passNumCond = num_genTriggerEventFlag_->off() || num_genTriggerEventFlag_->accept(iEvent, iSetup);
  int ls = iEvent.id().luminosityBlock();

  if (do_met_)
    metDQM_.fillHistograms(met, phi, ls, passNumCond);
  if (do_jet_)
    jetDQM_.fillHistograms(jets, pfmet, ls, passNumCond);
  if (do_ht_)
    htDQM_.fillHistograms(htjets, met, ls, passNumCond);
  if (do_hmg_)
    hmgDQM_.fillHistograms(photons, passedMesons, ls, passNumCond);
}

bool ObjMonitor::looseJetId(const double& abseta,
                            const double& NHF,
                            const double& NEMF,
                            const double& CHF,
                            const double& CEMF,
                            const unsigned& NumNeutralParticles,
                            const unsigned& CHM) {
  if (abseta <= 2.7) {
    unsigned NumConst = CHM + NumNeutralParticles;

    return ((NumConst > 1 && NHF < 0.99 && NEMF < 0.99) &&
            ((abseta <= 2.4 && CHF > 0 && CHM > 0 && CEMF < 0.99) || abseta > 2.4));
  } else if (abseta <= 3) {
    return (NumNeutralParticles > 2 && NEMF > 0.01 && NHF < 0.98);
  } else {
    return NumNeutralParticles > 10 && NEMF < 0.90;
  }
}
bool ObjMonitor::tightJetId(const double& abseta,
                            const double& NHF,
                            const double& NEMF,
                            const double& CHF,
                            const double& CEMF,
                            const unsigned& NumNeutralParticles,
                            const unsigned& CHM) {
  if (abseta <= 2.7) {
    unsigned NumConst = CHM + NumNeutralParticles;
    return (NumConst > 1 && NHF < 0.90 && NEMF < 0.90) &&
           ((abseta <= 2.4 && CHF > 0 && CHM > 0 && CEMF < 0.99) || abseta > 2.4);
  } else if (abseta <= 3) {
    return (NHF < 0.98 && NEMF > 0.01 && NumNeutralParticles > 2);
  } else {
    return (NEMF < 0.90 && NumNeutralParticles > 10);
  }
}

void ObjMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/OBJ");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("photons", edm::InputTag("gedPhotons"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("jetId", "");
  desc.add<std::string>("htjetSelection", "pt > 30");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<std::string>("phoSelection", "pt > 0");
  desc.add<std::string>("trkSelection", "pt > 0");
  desc.add<int>("njets", 0);
  desc.add<int>("nelectrons", 0);
  desc.add<int>("nmuons", 0);
  desc.add<int>("nphotons", 0);
  desc.add<int>("nmesons", 0);

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

  desc.add<bool>("doMETHistos", true);
  edm::ParameterSetDescription histoPSet;
  METDQM::fillMetDescription(histoPSet);
  desc.add<bool>("doJetHistos", true);
  JetDQM::fillJetDescription(histoPSet);
  desc.add<bool>("doHTHistos", true);
  HTDQM::fillHtDescription(histoPSet);
  desc.add<bool>("doHMesonGammaHistos", true);
  HMesonGammaDQM::fillHmgDescription(histoPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("objMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ObjMonitor);

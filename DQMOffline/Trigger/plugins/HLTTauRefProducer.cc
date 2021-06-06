#include "DQMOffline/Trigger/interface/HLTTauRefProducer.h"
#include "TLorentzVector.h"
// TAU includes
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
// ELECTRON includes
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
// MUON includes
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
//CaloTower includes
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "Math/GenVector/VectorUtil.h"

using namespace edm;
using namespace reco;
using namespace std;

HLTTauRefProducer::HLTTauRefProducer(const edm::ParameterSet& iConfig) {
  //One Parameter Set per Collection
  {
    auto const& pfTau = iConfig.getUntrackedParameter<edm::ParameterSet>("PFTaus");
    PFTaus_ = consumes<reco::PFTauCollection>(pfTau.getUntrackedParameter<InputTag>("PFTauProducer"));
    auto discs = pfTau.getUntrackedParameter<vector<InputTag>>("PFTauDiscriminators");
    auto discConts = pfTau.getUntrackedParameter<vector<InputTag>>("PFTauDiscriminatorContainers");
    PFTauDisContWPs_ = pfTau.getUntrackedParameter<vector<std::string>>("PFTauDiscriminatorContainerWPs");
    if (discConts.size() != PFTauDisContWPs_.size())
      throw cms::Exception("Configuration") << "HLTTauRefProducer: Input parameters PFTauDiscriminatorContainers and "
                                               "PFTauDiscriminatorContainerWPs must have the same number of entries!\n";
    for (auto const& tag : discs) {
      PFTauDis_.push_back(consumes<reco::PFTauDiscriminator>(tag));
    }
    for (auto const& tag : discConts) {
      PFTauDisCont_.push_back(consumes<reco::TauDiscriminatorContainer>(tag));
    }
    doPFTaus_ = pfTau.getUntrackedParameter<bool>("doPFTaus", false);
    ptMinPFTau_ = pfTau.getUntrackedParameter<double>("ptMin", 15.);
    etaMinPFTau_ = pfTau.getUntrackedParameter<double>("etaMin", -2.5);
    etaMaxPFTau_ = pfTau.getUntrackedParameter<double>("etaMax", 2.5);
    phiMinPFTau_ = pfTau.getUntrackedParameter<double>("phiMin", -3.15);
    phiMaxPFTau_ = pfTau.getUntrackedParameter<double>("phiMax", 3.15);
  }

  {
    auto const& electrons = iConfig.getUntrackedParameter<edm::ParameterSet>("Electrons");
    Electrons_ = consumes<reco::GsfElectronCollection>(electrons.getUntrackedParameter<InputTag>("ElectronCollection"));
    doElectrons_ = electrons.getUntrackedParameter<bool>("doElectrons", false);
    e_ctfTrackCollectionSrc_ = electrons.getUntrackedParameter<InputTag>("TrackCollection");
    e_ctfTrackCollection_ = consumes<reco::TrackCollection>(e_ctfTrackCollectionSrc_);
    ptMinElectron_ = electrons.getUntrackedParameter<double>("ptMin", 15.);
    e_doTrackIso_ = electrons.getUntrackedParameter<bool>("doTrackIso", false);
    e_trackMinPt_ = electrons.getUntrackedParameter<double>("ptMinTrack", 1.5);
    e_lipCut_ = electrons.getUntrackedParameter<double>("lipMinTrack", 1.5);
    e_minIsoDR_ = electrons.getUntrackedParameter<double>("InnerConeDR", 0.02);
    e_maxIsoDR_ = electrons.getUntrackedParameter<double>("OuterConeDR", 0.6);
    e_isoMaxSumPt_ = electrons.getUntrackedParameter<double>("MaxIsoVar", 0.02);
  }

  {
    auto const& muons = iConfig.getUntrackedParameter<edm::ParameterSet>("Muons");
    Muons_ = consumes<reco::MuonCollection>(muons.getUntrackedParameter<InputTag>("MuonCollection"));
    doMuons_ = muons.getUntrackedParameter<bool>("doMuons", false);
    ptMinMuon_ = muons.getUntrackedParameter<double>("ptMin", 15.);
  }

  {
    auto const& jets = iConfig.getUntrackedParameter<edm::ParameterSet>("Jets");
    Jets_ = consumes<reco::CaloJetCollection>(jets.getUntrackedParameter<InputTag>("JetCollection"));
    doJets_ = jets.getUntrackedParameter<bool>("doJets");
    ptMinJet_ = jets.getUntrackedParameter<double>("etMin");
  }

  {
    auto const& towers = iConfig.getUntrackedParameter<edm::ParameterSet>("Towers");
    Towers_ = consumes<CaloTowerCollection>(towers.getUntrackedParameter<InputTag>("TowerCollection"));
    doTowers_ = towers.getUntrackedParameter<bool>("doTowers");
    ptMinTower_ = towers.getUntrackedParameter<double>("etMin");
    towerIsol_ = towers.getUntrackedParameter<double>("towerIsolation");
  }

  {
    auto const& photons = iConfig.getUntrackedParameter<edm::ParameterSet>("Photons");
    Photons_ = consumes<reco::PhotonCollection>(photons.getUntrackedParameter<InputTag>("PhotonCollection"));
    doPhotons_ = photons.getUntrackedParameter<bool>("doPhotons");
    ptMinPhoton_ = photons.getUntrackedParameter<double>("etMin");
    photonEcalIso_ = photons.getUntrackedParameter<double>("ECALIso");
  }

  {
    auto const& met = iConfig.getUntrackedParameter<edm::ParameterSet>("MET");
    MET_ = consumes<reco::CaloMETCollection>(met.getUntrackedParameter<InputTag>("METCollection"));
    doMET_ = met.getUntrackedParameter<bool>("doMET", false);
    ptMinMET_ = met.getUntrackedParameter<double>("ptMin", 15.);
  }

  etaMin_ = iConfig.getUntrackedParameter<double>("EtaMin", -2.5);
  etaMax_ = iConfig.getUntrackedParameter<double>("EtaMax", 2.5);
  phiMin_ = iConfig.getUntrackedParameter<double>("PhiMin", -3.15);
  phiMax_ = iConfig.getUntrackedParameter<double>("PhiMax", 3.15);

  //recoCollections
  produces<LorentzVectorCollection>("PFTaus");
  produces<LorentzVectorCollection>("Electrons");
  produces<LorentzVectorCollection>("Muons");
  produces<LorentzVectorCollection>("Jets");
  produces<LorentzVectorCollection>("Photons");
  produces<LorentzVectorCollection>("Towers");
  produces<LorentzVectorCollection>("MET");
}

void HLTTauRefProducer::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const {
  if (doPFTaus_)
    doPFTaus(iID, iEvent);
  if (doElectrons_)
    doElectrons(iEvent);
  if (doMuons_)
    doMuons(iEvent);
  if (doJets_)
    doJets(iEvent);
  if (doPhotons_)
    doPhotons(iEvent);
  if (doTowers_)
    doTowers(iEvent);
  if (doMET_)
    doMET(iEvent);
}

void HLTTauRefProducer::doPFTaus(edm::StreamID iID, edm::Event& iEvent) const {
  auto product_PFTaus = make_unique<LorentzVectorCollection>();

  edm::Handle<PFTauCollection> pftaus;
  if (iEvent.getByToken(PFTaus_, pftaus)) {
    // Retrieve ID container indices if config history changes, in particular for the first event.
    if (streamCache(iID)->first != iEvent.processHistoryID()) {
      streamCache(iID)->first = iEvent.processHistoryID();
      streamCache(iID)->second.resize(PFTauDisContWPs_.size());
      for (size_t i = 0; i < PFTauDisCont_.size(); ++i) {
        auto const aHandle = iEvent.getHandle(PFTauDisCont_[i]);
        auto const aProv = aHandle.provenance();
        if (aProv == nullptr)
          aHandle.whyFailed()->raise();
        const auto& psetsFromProvenance = edm::parameterSet(aProv->stable(), iEvent.processHistory());
        if (psetsFromProvenance.exists("workingPoints")) {
          auto const idlist = psetsFromProvenance.getParameter<std::vector<std::string>>("workingPoints");
          bool found = false;
          for (size_t j = 0; j < idlist.size(); ++j) {
            if (PFTauDisContWPs_[i] == idlist[j]) {
              found = true;
              streamCache(iID)->second[i] = j;
            }
          }
          if (!found)
            throw cms::Exception("Configuration")
                << "HLTTauRefProducer: Requested working point '" << PFTauDisContWPs_[i] << "' not found!\n";
        } else if (psetsFromProvenance.exists("IDWPdefinitions")) {
          auto const idlist = psetsFromProvenance.getParameter<std::vector<edm::ParameterSet>>("IDWPdefinitions");
          bool found = false;
          for (size_t j = 0; j < idlist.size(); ++j) {
            if (PFTauDisContWPs_[i] == idlist[j].getParameter<std::string>("IDname")) {
              found = true;
              streamCache(iID)->second[i] = j;
            }
          }
          if (!found)
            throw cms::Exception("Configuration")
                << "HLTTauRefProducer: Requested working point '" << PFTauDisContWPs_[i] << "' not found!\n";
        } else
          throw cms::Exception("Configuration")
              << "HLTTauRefProducer: No suitable ID list found in provenace config!\n";
      }
    }
    for (unsigned int i = 0; i < pftaus->size(); ++i) {
      auto const& pftau = (*pftaus)[i];
      if (pftau.pt() > ptMinPFTau_ && pftau.eta() > etaMinPFTau_ && pftau.eta() < etaMaxPFTau_ &&
          pftau.phi() > phiMinPFTau_ && pftau.phi() < phiMaxPFTau_) {
        reco::PFTauRef thePFTau{pftaus, i};
        bool passAll{true};

        for (auto const& token : PFTauDis_) {
          edm::Handle<reco::PFTauDiscriminator> pftaudis;
          if (iEvent.getByToken(token, pftaudis)) {
            if ((*pftaudis)[thePFTau] < 0.5) {
              passAll = false;
              break;
            }
          } else {
            passAll = false;
            break;
          }
        }

        int idx = 0;
        for (auto const& token : PFTauDisCont_) {
          edm::Handle<reco::TauDiscriminatorContainer> pftaudis;
          if (iEvent.getByToken(token, pftaudis)) {
            //WP vector not filled if prediscriminator in RecoTauDiscriminator failed.
            if ((*pftaudis)[thePFTau].workingPoints.empty() ||
                !(*pftaudis)[thePFTau].workingPoints.at(streamCache(iID)->second[idx])) {
              passAll = false;
              break;
            }
          } else {
            passAll = false;
            break;
          }
          idx++;
        }
        if (passAll) {
          product_PFTaus->emplace_back(pftau.px(), pftau.py(), pftau.pz(), pftau.energy());
        }
      }
    }
  }
  iEvent.put(move(product_PFTaus), "PFTaus");
}

void HLTTauRefProducer::doElectrons(edm::Event& iEvent) const {
  auto product_Electrons = make_unique<LorentzVectorCollection>();

  edm::Handle<reco::TrackCollection> pCtfTracks;
  if (!iEvent.getByToken(e_ctfTrackCollection_, pCtfTracks)) {
    edm::LogInfo("") << "Error! Can't get " << e_ctfTrackCollectionSrc_.label() << " by label. ";
    iEvent.put(move(product_Electrons), "Electrons");
    return;
  }

  edm::Handle<GsfElectronCollection> electrons;
  if (iEvent.getByToken(Electrons_, electrons)) {
    for (size_t i = 0; i < electrons->size(); ++i) {
      edm::Ref<reco::GsfElectronCollection> electronRef(electrons, i);
      auto const& electron = (*electrons)[i];
      if (electron.pt() > ptMinElectron_ && fabs(electron.eta()) < etaMax_) {
        if (e_doTrackIso_) {
          double sum_of_pt_ele{};
          for (auto const& tr : *pCtfTracks) {
            double const lip{electron.gsfTrack()->dz() - tr.dz()};
            if (tr.pt() > e_trackMinPt_ && fabs(lip) < e_lipCut_) {
              double dphi{fabs(tr.phi() - electron.trackMomentumAtVtx().phi())};
              if (dphi > acos(-1.)) {
                dphi = 2 * acos(-1.) - dphi;
              }
              double const deta{fabs(tr.eta() - electron.trackMomentumAtVtx().eta())};
              double const dr_ctf_ele{sqrt(deta * deta + dphi * dphi)};
              if ((dr_ctf_ele > e_minIsoDR_) && (dr_ctf_ele < e_maxIsoDR_)) {
                double const cft_pt_2{tr.pt() * tr.pt()};
                sum_of_pt_ele += cft_pt_2;
              }
            }
          }
          double const isolation_value_ele{sum_of_pt_ele /
                                           (electron.trackMomentumAtVtx().Rho() * electron.trackMomentumAtVtx().Rho())};
          if (isolation_value_ele < e_isoMaxSumPt_) {
            product_Electrons->emplace_back(electron.px(), electron.py(), electron.pz(), electron.energy());
          }
        } else {
          product_Electrons->emplace_back(electron.px(), electron.py(), electron.pz(), electron.energy());
        }
      }
    }
  }
  iEvent.put(move(product_Electrons), "Electrons");
}

void HLTTauRefProducer::doMuons(edm::Event& iEvent) const {
  auto product_Muons = make_unique<LorentzVectorCollection>();

  edm::Handle<MuonCollection> muons;
  if (iEvent.getByToken(Muons_, muons)) {
    for (auto const& muon : *muons) {
      if (muon.pt() > ptMinMuon_ && muon.eta() > etaMin_ && muon.eta() < etaMax_ && muon.phi() > phiMin_ &&
          muon.phi() < phiMax_) {
        product_Muons->emplace_back(muon.px(), muon.py(), muon.pz(), muon.energy());
      }
    }
  }
  iEvent.put(move(product_Muons), "Muons");
}

void HLTTauRefProducer::doJets(edm::Event& iEvent) const {
  auto product_Jets = make_unique<LorentzVectorCollection>();

  edm::Handle<CaloJetCollection> jets;
  if (iEvent.getByToken(Jets_, jets)) {
    for (auto const& jet : *jets) {
      if (jet.et() > ptMinJet_ && jet.eta() > etaMin_ && jet.eta() < etaMax_ && jet.phi() > phiMin_ &&
          jet.phi() < phiMax_) {
        product_Jets->emplace_back(jet.px(), jet.py(), jet.pz(), jet.energy());
      }
    }
  }
  iEvent.put(move(product_Jets), "Jets");
}

void HLTTauRefProducer::doTowers(edm::Event& iEvent) const {
  auto product_Towers = make_unique<LorentzVectorCollection>();

  edm::Handle<CaloTowerCollection> towers;
  if (iEvent.getByToken(Towers_, towers)) {
    for (auto const& tower1 : *towers) {
      if (tower1.pt() > ptMinTower_ && tower1.eta() > etaMin_ && tower1.eta() < etaMax_ && tower1.phi() > phiMin_ &&
          tower1.phi() < phiMax_) {
        //calculate isolation
        double isolET{};
        for (auto const& tower2 : *towers) {
          if (ROOT::Math::VectorUtil::DeltaR(tower1.p4(), tower2.p4()) < 0.5) {
            isolET += tower2.pt();
          }
          isolET -= tower1.pt();
        }
        if (isolET < towerIsol_) {
          product_Towers->emplace_back(tower1.px(), tower1.py(), tower1.pz(), tower1.energy());
        }
      }
    }
  }
  iEvent.put(move(product_Towers), "Towers");
}

void HLTTauRefProducer::doPhotons(edm::Event& iEvent) const {
  auto product_Gammas = make_unique<LorentzVectorCollection>();

  edm::Handle<PhotonCollection> photons;
  if (iEvent.getByToken(Photons_, photons)) {
    for (auto const& photon : *photons) {
      if (photon.ecalRecHitSumEtConeDR04() < photonEcalIso_ && photon.et() > ptMinPhoton_ && photon.eta() > etaMin_ &&
          photon.eta() < etaMax_ && photon.phi() > phiMin_ && photon.phi() < phiMax_) {
        product_Gammas->emplace_back(photon.px(), photon.py(), photon.pz(), photon.energy());
      }
    }
  }
  iEvent.put(move(product_Gammas), "Photons");
}

void HLTTauRefProducer::doMET(edm::Event& iEvent) const {
  auto product_MET = make_unique<LorentzVectorCollection>();

  edm::Handle<reco::CaloMETCollection> met;
  if (iEvent.getByToken(MET_, met) && !met->empty()) {
    auto const& metMom = met->front().p4();
    product_MET->emplace_back(metMom.Px(), metMom.Py(), 0, metMom.Pt());
  }
  iEvent.put(move(product_MET), "MET");
}

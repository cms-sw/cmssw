#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "CommonTools/ParticleFlow/test/PFIsoReaderDemo.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

PFIsoReaderDemo::PFIsoReaderDemo(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);
  inputTagGsfElectrons_ = iConfig.getParameter<edm::InputTag>("Electrons");
  tokenGsfElectrons_ = consumes<reco::GsfElectronCollection>(inputTagGsfElectrons_);
  inputTagPhotons_ = iConfig.getParameter<edm::InputTag>("Photons");
  tokenPhotons_ = consumes<reco::PhotonCollection>(inputTagPhotons_);

  //   // not needed at the moment
  //   inputTagPFCandidateMap_ = iConfig.getParameter< edm::InputTag>("PFCandidateMap");

  //   inputTagIsoDepElectrons_ = iConfig.getParameter< std::vector<edm::InputTag> >("IsoDepElectron");
  tokensIsoDepElectrons_ = edm::vector_transform(
      iConfig.getParameter<std::vector<edm::InputTag> >("IsoDepElectron"),
      [this](edm::InputTag const& tag) { return consumes<edm::ValueMap<reco::IsoDeposit> >(tag); });
  //   inputTagIsoDepPhotons_ = iConfig.getParameter< std::vector<edm::InputTag> >("IsoDepPhoton");
  tokensIsoDepPhotons_ = edm::vector_transform(
      iConfig.getParameter<std::vector<edm::InputTag> >("IsoDepPhoton"),
      [this](edm::InputTag const& tag) { return consumes<edm::ValueMap<reco::IsoDeposit> >(tag); });
  // No longer needed. e/g recommendation (04/04/12)
  //  inputTagIsoValElectronsNoPFId_ = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValElectronNoPF");
  //   inputTagIsoValElectronsPFId_   = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValElectronPF");
  tokensIsoValElectronsPFId_ =
      edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("IsoValElectronPF"),
                            [this](edm::InputTag const& tag) { return consumes<edm::ValueMap<double> >(tag); });
  //   inputTagIsoValPhotonsPFId_   = iConfig.getParameter< std::vector<edm::InputTag> >("IsoValPhoton");
  tokensIsoValPhotonsPFId_ =
      edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("IsoValPhoton"),
                            [this](edm::InputTag const& tag) { return consumes<edm::ValueMap<double> >(tag); });

  printElectrons_ = iConfig.getParameter<bool>("PrintElectrons");
  printPhotons_ = iConfig.getParameter<bool>("PrintPhotons");

  // Control plots
  TFileDirectory dir = fileservice_->mkdir("PF ISO");
  chargedBarrelElectrons_ = dir.make<TH1F>("chargedBarrelElectrons", ";Sum pT/pT", 100, 0, 4);
  photonBarrelElectrons_ = dir.make<TH1F>("photonBarrelElectrons", ";Sum pT/pT", 100, 0, 4);
  neutralBarrelElectrons_ = dir.make<TH1F>("neutralBarrelElectrons", ";Sum pT/pT", 100, 0, 4);

  chargedEndcapsElectrons_ = dir.make<TH1F>("chargedEndcapsElectrons", ";Sum pT/pT", 100, 0, 4);
  photonEndcapsElectrons_ = dir.make<TH1F>("photonEndcapsElectrons", ";Sum pT/pT", 100, 0, 4);
  neutralEndcapsElectrons_ = dir.make<TH1F>("neutralEndcapsElectrons", ";Sum pT/pT", 100, 0, 4);

  sumBarrelElectrons_ = dir.make<TH1F>("allbarrelElectrons", ";Sum pT/pT", 100, 0, 4);
  sumEndcapsElectrons_ = dir.make<TH1F>("allendcapsElectrons", ";Sum pT/pT", 100, 0, 4);

  chargedBarrelPhotons_ = dir.make<TH1F>("chargedBarrelPhotons", ";Sum pT/pT", 100, 0, 4);
  photonBarrelPhotons_ = dir.make<TH1F>("photonBarrelPhotons", ";Sum pT/pT", 100, 0, 4);
  neutralBarrelPhotons_ = dir.make<TH1F>("neutralBarrelPhotons", ";Sum pT/pT", 100, 0, 4);

  chargedEndcapsPhotons_ = dir.make<TH1F>("chargedEndcapsPhotons", ";Sum pT/pT", 100, 0, 4);
  photonEndcapsPhotons_ = dir.make<TH1F>("photonEndcapsPhotons", ";Sum pT/pT", 100, 0, 4);
  neutralEndcapsPhotons_ = dir.make<TH1F>("neutralEndcapsPhotons", ";Sum pT/pT", 100, 0, 4);

  sumBarrelPhotons_ = dir.make<TH1F>("allbarrelPhotons", ";Sum pT/pT", 100, 0, 4);
  sumEndcapsPhotons_ = dir.make<TH1F>("allendcapsPhotons", ";Sum pT/pT", 100, 0, 4);
}

PFIsoReaderDemo::~PFIsoReaderDemo() { ; }

void PFIsoReaderDemo::beginRun(edm::Run const&, edm::EventSetup const&) { ; }

void PFIsoReaderDemo::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {
  edm::Handle<reco::GsfElectronCollection> gsfElectronH;
  bool found = iEvent.getByToken(tokenGsfElectrons_, gsfElectronH);
  if (!found) {
    std::ostringstream err;
    err << " cannot get GsfElectrons: " << inputTagGsfElectrons_ << std::endl;
    edm::LogError("PFIsoReaderDemo") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  edm::Handle<reco::PhotonCollection> photonH;
  found = iEvent.getByToken(tokenPhotons_, photonH);
  if (!found) {
    std::ostringstream err;
    err << " cannot get Photons: " << inputTagPhotons_ << std::endl;
    edm::LogError("PFIsoReaderDemo") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  // get the iso deposits. 3 (charged hadrons, photons, neutral hadrons)
  unsigned nTypes = 3;
  IsoDepositMaps electronIsoDep(nTypes);

  for (size_t j = 0; j < tokensIsoDepElectrons_.size(); ++j) {
    iEvent.getByToken(tokensIsoDepElectrons_[j], electronIsoDep[j]);
  }

  IsoDepositMaps photonIsoDep(nTypes);
  for (size_t j = 0; j < tokensIsoDepPhotons_.size(); ++j) {
    iEvent.getByToken(tokensIsoDepPhotons_[j], photonIsoDep[j]);
  }

  IsoDepositVals electronIsoValPFId(nTypes);
  IsoDepositVals photonIsoValPFId(nTypes);
  // just renaming
  const IsoDepositVals* electronIsoVals = &electronIsoValPFId;

  for (size_t j = 0; j < tokensIsoValElectronsPFId_.size(); ++j) {
    iEvent.getByToken(tokensIsoValElectronsPFId_[j], electronIsoValPFId[j]);
  }

  for (size_t j = 0; j < tokensIsoValPhotonsPFId_.size(); ++j) {
    iEvent.getByToken(tokensIsoValPhotonsPFId_[j], photonIsoValPFId[j]);
  }

  // Electrons - from reco
  if (printElectrons_) {
    unsigned nele = gsfElectronH->size();

    for (unsigned iele = 0; iele < nele; ++iele) {
      reco::GsfElectronRef myElectronRef(gsfElectronH, iele);

      double charged = (*(*electronIsoVals)[0])[myElectronRef];
      double photon = (*(*electronIsoVals)[1])[myElectronRef];
      double neutral = (*(*electronIsoVals)[2])[myElectronRef];

      std::cout << "Electron: "
                << " run " << iEvent.id().run() << " lumi " << iEvent.id().luminosityBlock() << " event "
                << iEvent.id().event();
      std::cout << " pt " << myElectronRef->pt() << " eta " << myElectronRef->eta() << " phi " << myElectronRef->phi()
                << " charge " << myElectronRef->charge() << " : ";
      std::cout << " ChargedIso " << charged;
      std::cout << " PhotonIso " << photon;
      std::cout << " NeutralHadron Iso " << neutral << std::endl;

      if (myElectronRef->isEB()) {
        chargedBarrelElectrons_->Fill(charged / myElectronRef->pt());
        photonBarrelElectrons_->Fill(photon / myElectronRef->pt());
        neutralBarrelElectrons_->Fill(neutral / myElectronRef->pt());
        sumBarrelElectrons_->Fill((charged + photon + neutral) / myElectronRef->pt());
      } else {
        chargedEndcapsElectrons_->Fill(charged / myElectronRef->pt());
        photonEndcapsElectrons_->Fill(photon / myElectronRef->pt());
        neutralEndcapsElectrons_->Fill(neutral / myElectronRef->pt());
        sumEndcapsElectrons_->Fill((charged + photon + neutral) / myElectronRef->pt());
      }
    }
  }

  // Photons - from reco
  const IsoDepositVals* photonIsoVals = &photonIsoValPFId;

  if (printPhotons_) {
    unsigned npho = photonH->size();

    for (unsigned ipho = 0; ipho < npho; ++ipho) {
      reco::PhotonRef myPhotonRef(photonH, ipho);

      double charged = (*(*photonIsoVals)[0])[myPhotonRef];
      double photon = (*(*photonIsoVals)[1])[myPhotonRef];
      double neutral = (*(*photonIsoVals)[2])[myPhotonRef];

      std::cout << "Photon: "
                << " run " << iEvent.id().run() << " lumi " << iEvent.id().luminosityBlock() << " event "
                << iEvent.id().event();
      std::cout << " pt " << myPhotonRef->pt() << " eta " << myPhotonRef->eta() << " phi " << myPhotonRef->phi()
                << " charge " << myPhotonRef->charge() << " : ";
      std::cout << " ChargedIso " << charged;
      std::cout << " PhotonIso " << photon;
      std::cout << " NeutralHadron Iso " << neutral << std::endl;

      if (myPhotonRef->isEB()) {
        chargedBarrelPhotons_->Fill(charged / myPhotonRef->pt());
        photonBarrelPhotons_->Fill(photon / myPhotonRef->pt());
        neutralBarrelPhotons_->Fill(neutral / myPhotonRef->pt());
        sumBarrelPhotons_->Fill((charged + photon + neutral) / myPhotonRef->pt());
      } else {
        chargedEndcapsPhotons_->Fill(charged / myPhotonRef->pt());
        photonEndcapsPhotons_->Fill(photon / myPhotonRef->pt());
        neutralEndcapsPhotons_->Fill(neutral / myPhotonRef->pt());
        sumEndcapsPhotons_->Fill((charged + photon + neutral) / myPhotonRef->pt());
      }
    }
  }
}

DEFINE_FWK_MODULE(PFIsoReaderDemo);

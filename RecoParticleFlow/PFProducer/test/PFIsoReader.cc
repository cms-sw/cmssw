#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "RecoParticleFlow/PFProducer/test/PFIsoReader.h"

PFIsoReader::PFIsoReader(const edm::ParameterSet& iConfig)
    : inputTagGsfElectrons_(iConfig.getParameter<edm::InputTag>("Electrons")),
      inputTagPhotons_(iConfig.getParameter<edm::InputTag>("Photons")),
      inputTagPFCandidates_(iConfig.getParameter<edm::InputTag>("PFCandidates")),
      inputTagValueMapPhotons_(iConfig.getParameter<edm::InputTag>("PhotonValueMap")),
      inputTagValueMapElectrons_(iConfig.getParameter<edm::InputTag>("ElectronValueMap")),
      inputTagValueMapMerged_(iConfig.getParameter<edm::InputTag>("MergedValueMap")),
      inputTagElectronIsoDeposits_(iConfig.getParameter<std::vector<edm::InputTag> >("ElectronIsoDeposits")),
      inputTagPhotonIsoDeposits_(iConfig.getParameter<std::vector<edm::InputTag> >("PhotonIsoDeposits")),
      useValueMaps_(iConfig.getParameter<bool>("useEGPFValueMaps")),
      pfCandToken_(consumes<reco::PFCandidateCollection>(inputTagPFCandidates_)),
      elecToken_(consumes<reco::GsfElectronCollection>(inputTagGsfElectrons_)),
      photonToken_(consumes<reco::PhotonCollection>(inputTagPhotons_)),
      elecMapToken_(consumes<edm::ValueMap<reco::PFCandidatePtr> >(inputTagValueMapElectrons_)),
      photonMapToken_(consumes<edm::ValueMap<reco::PFCandidatePtr> >(inputTagValueMapPhotons_)),
      mergeMapToken_(consumes<edm::ValueMap<reco::PFCandidatePtr> >(inputTagValueMapMerged_)) {
  for (auto const& tag : inputTagElectronIsoDeposits_)
    isoElecToken_.emplace_back(consumes<edm::Handle<edm::ValueMap<reco::IsoDeposit> > >(tag));
  for (auto const& tag : inputTagPhotonIsoDeposits_)
    isoPhotToken_.emplace_back(consumes<edm::Handle<edm::ValueMap<reco::IsoDeposit> > >(tag));
}

void PFIsoReader::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {
  const edm::Handle<reco::PFCandidateCollection>& pfCandidatesH = iEvent.getHandle(pfCandToken_);
  if (!pfCandidatesH.isValid()) {
    std::ostringstream err;
    err << " cannot get PFCandidates: " << inputTagPFCandidates_;
    edm::LogError("PFIsoReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  const edm::Handle<reco::GsfElectronCollection>& gsfElectronH = iEvent.getHandle(elecToken_);
  if (!gsfElectronH.isValid()) {
    std::ostringstream err;
    err << " cannot get GsfElectrons: " << inputTagGsfElectrons_;
    edm::LogError("PFIsoReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  const edm::Handle<reco::PhotonCollection>& photonH = iEvent.getHandle(photonToken_);
  if (!photonH.isValid()) {
    std::ostringstream err;
    err << " cannot get Photonss: " << inputTagPhotons_;
    edm::LogError("PFIsoReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  // Get the value maps

  const edm::Handle<edm::ValueMap<reco::PFCandidatePtr> >& electronValMapH = iEvent.getHandle(elecMapToken_);
  const edm::ValueMap<reco::PFCandidatePtr>& myElectronValMap(*electronValMapH);

  edm::LogVerbatim("PFIsoReader") << " Read Electron Value Map " << myElectronValMap.size();

  //  const edm::Handle<edm::ValueMap<reco::PFCandidatePtr> >& photonValMapH nd = iEvent.getHandle(photonMapToken_);
  //   const edm::ValueMap<reco::PFCandidatePtr> & myPhotonValMap(*photonValMapH);

  const edm::Handle<edm::ValueMap<reco::PFCandidatePtr> >& mergedValMapH = iEvent.getHandle(mergeMapToken_);
  const edm::ValueMap<reco::PFCandidatePtr>& myMergedValMap(*mergedValMapH);

  // get the iso deposits
  IsoDepositMaps electronIsoDep(inputTagElectronIsoDeposits_.size());
  IsoDepositMaps photonIsoDep(inputTagPhotonIsoDeposits_.size());

  for (size_t j = 0; j < inputTagElectronIsoDeposits_.size(); ++j)
    iEvent.getByToken(isoElecToken_[j], electronIsoDep[j]);
  for (size_t j = 0; j < inputTagPhotonIsoDeposits_.size(); ++j)
    iEvent.getByToken(isoPhotToken_[j], photonIsoDep[j]);

  // Photons - from reco
  unsigned nphot = photonH->size();
  edm::LogVerbatim("PFIsoReader") << "Photon: " << nphot;
  for (unsigned iphot = 0; iphot < nphot; ++iphot) {
    reco::PhotonRef myPhotRef(photonH, iphot);
    //    const reco::PFCandidatePtr & pfPhotPtr(myPhotonValMap[myPhotRef]);
    const reco::PFCandidatePtr& pfPhotPtr(myMergedValMap[myPhotRef]);
    printIsoDeposits(photonIsoDep, pfPhotPtr);
  }

  // Photons - from PF Candidates
  unsigned ncandidates = pfCandidatesH->size();
  edm::LogVerbatim("PFIsoReader") << "Candidates: " << ncandidates;
  for (unsigned icand = 0; icand < ncandidates; ++icand) {
    const reco::PFCandidate& cand((*pfCandidatesH)[icand]);
    //    edm::LogVerbatim("PFIsoReader") << " Pdg " << cand.pdgId() << " mva " << cand.mva_nothing_gamma();
    if (!(cand.pdgId() == 22 && cand.mva_nothing_gamma() > 0))
      continue;

    reco::PFCandidatePtr myPFCandidatePtr(pfCandidatesH, icand);
    printIsoDeposits(photonIsoDep, myPFCandidatePtr);
  }

  // Electrons - from reco
  unsigned nele = gsfElectronH->size();
  edm::LogVerbatim("PFIsoReader") << "Electron: " << nele;
  for (unsigned iele = 0; iele < nele; ++iele) {
    reco::GsfElectronRef myElectronRef(gsfElectronH, iele);

    if (myElectronRef->mva_e_pi() < -1)
      continue;
    //const reco::PFCandidatePtr & pfElePtr(myElectronValMap[myElectronRef]);
    const reco::PFCandidatePtr pfElePtr(myElectronValMap[myElectronRef]);
    printIsoDeposits(electronIsoDep, pfElePtr);
  }

  // Electrons - from PFCandidate
  nele = gsfElectronH->size();
  edm::LogVerbatim("PFIsoReader") << "Candidates: " << nele;
  for (unsigned icand = 0; icand < ncandidates; ++icand) {
    const reco::PFCandidate& cand((*pfCandidatesH)[icand]);

    if (!(abs(cand.pdgId()) == 11))
      continue;

    reco::PFCandidatePtr myPFCandidatePtr(pfCandidatesH, icand);
    printIsoDeposits(electronIsoDep, myPFCandidatePtr);
  }
}

void PFIsoReader::printIsoDeposits(const IsoDepositMaps& isodepmap, const reco::PFCandidatePtr& ptr) const {
  edm::LogVerbatim("PFIsoReader") << " Isodeposits for " << ptr.id() << " " << ptr.key();
  unsigned nIsoDepTypes = isodepmap.size();  // should be 3 (charged hadrons, photons, neutral hadrons)
  for (unsigned ideptype = 0; ideptype < nIsoDepTypes; ++ideptype) {
    const reco::IsoDeposit& isoDep((*isodepmap[ideptype])[ptr]);
    typedef reco::IsoDeposit::const_iterator IM;
    edm::LogVerbatim("PFIsoReader") << " Iso deposits type " << ideptype;
    for (IM im = isoDep.begin(); im != isoDep.end(); ++im) {
      edm::LogVerbatim("PFIsoReader") << "dR " << im->dR() << " val " << im->value();
    }
  }
}

DEFINE_FWK_MODULE(PFIsoReader);

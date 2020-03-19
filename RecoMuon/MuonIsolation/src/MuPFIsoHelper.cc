#include "RecoMuon/MuonIsolation/interface/MuPFIsoHelper.h"

MuPFIsoHelper::MuPFIsoHelper(const std::map<std::string, edm::ParameterSet>& labelMap, edm::ConsumesCollector&& iC)
    : labelMap_(labelMap) {
  edm::Handle<CandDoubleMap> nullHandle;
  for (std::map<std::string, edm::ParameterSet>::const_iterator i = labelMap_.begin(); i != labelMap_.end(); ++i) {
    chargedParticle_.push_back(nullHandle);
    chargedHadron_.push_back(nullHandle);
    neutralHadron_.push_back(nullHandle);
    neutralHadronHighThreshold_.push_back(nullHandle);
    photon_.push_back(nullHandle);
    photonHighThreshold_.push_back(nullHandle);
    pu_.push_back(nullHandle);

    chargedParticleToken_.push_back(
        iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("chargedParticle")));
    chargedHadronToken_.push_back(iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("chargedHadron")));
    neutralHadronToken_.push_back(iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("neutralHadron")));
    neutralHadronHighThresholdToken_.push_back(
        iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("neutralHadronHighThreshold")));
    photonToken_.push_back(iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("photon")));
    photonHighThresholdToken_.push_back(
        iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("photonHighThreshold")));
    puToken_.push_back(iC.consumes<CandDoubleMap>(i->second.getParameter<edm::InputTag>("pu")));
  }
}

MuPFIsoHelper::~MuPFIsoHelper() {}

reco::MuonPFIsolation MuPFIsoHelper::makeIsoDeposit(reco::MuonRef& muonRef,
                                                    const edm::Handle<CandDoubleMap>& chargedParticle,
                                                    const edm::Handle<CandDoubleMap>& chargedHadron,
                                                    const edm::Handle<CandDoubleMap>& neutralHadron,
                                                    const edm::Handle<CandDoubleMap>& neutralHadronHighThreshold,
                                                    const edm::Handle<CandDoubleMap>& photon,
                                                    const edm::Handle<CandDoubleMap>& photonHighThreshold,
                                                    const edm::Handle<CandDoubleMap>& pu) {
  reco::MuonPFIsolation iso;
  if (chargedParticle.isValid())
    iso.sumChargedParticlePt = (*chargedParticle)[muonRef];

  if (chargedHadron.isValid())
    iso.sumChargedHadronPt = (*chargedHadron)[muonRef];

  if (neutralHadron.isValid())
    iso.sumNeutralHadronEt = (*neutralHadron)[muonRef];

  if (neutralHadronHighThreshold.isValid())
    iso.sumNeutralHadronEtHighThreshold = (*neutralHadronHighThreshold)[muonRef];

  if (photon.isValid())
    iso.sumPhotonEt = (*photon)[muonRef];

  if (photonHighThreshold.isValid())
    iso.sumPhotonEtHighThreshold = (*photonHighThreshold)[muonRef];

  if (pu.isValid())
    iso.sumPUPt = (*pu)[muonRef];

  return iso;
}

int MuPFIsoHelper::embedPFIsolation(reco::Muon& muon, reco::MuonRef& muonRef) {
  unsigned int count = 0;
  for (std::map<std::string, edm::ParameterSet>::const_iterator i = labelMap_.begin(); i != labelMap_.end(); ++i) {
    reco::MuonPFIsolation iso = makeIsoDeposit(muonRef,
                                               chargedParticle_[count],
                                               chargedHadron_[count],
                                               neutralHadron_[count],
                                               neutralHadronHighThreshold_[count],
                                               photon_[count],
                                               photonHighThreshold_[count],
                                               pu_[count]);

    muon.setPFIsolation(i->first, iso);
    count++;
  }

  return 0;
}

void MuPFIsoHelper::beginEvent(const edm::Event& iEvent) {
  unsigned int count = 0;
  for (std::map<std::string, edm::ParameterSet>::const_iterator i = labelMap_.begin(); i != labelMap_.end(); ++i) {
    iEvent.getByToken(chargedParticleToken_[count], chargedParticle_[count]);
    iEvent.getByToken(chargedHadronToken_[count], chargedHadron_[count]);
    iEvent.getByToken(neutralHadronToken_[count], neutralHadron_[count]);
    iEvent.getByToken(neutralHadronHighThresholdToken_[count], neutralHadronHighThreshold_[count]);
    iEvent.getByToken(photonToken_[count], photon_[count]);
    iEvent.getByToken(photonHighThresholdToken_[count], photonHighThreshold_[count]);
    iEvent.getByToken(puToken_[count], pu_[count]);
    count++;
  }
}

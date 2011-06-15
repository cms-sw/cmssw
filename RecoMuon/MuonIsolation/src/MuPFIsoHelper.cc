#include "RecoMuon/MuonIsolation/interface/MuPFIsoHelper.h"


MuPFIsoHelper::MuPFIsoHelper(const edm::ParameterSet& iConfig):
  isoCfg03_(iConfig.getParameter<edm::ParameterSet>("isolationR03")),
  isoCfg04_(iConfig.getParameter<edm::ParameterSet>("isolationR04"))
{
  

}



MuPFIsoHelper::~MuPFIsoHelper() {

}



int MuPFIsoHelper::embedPFIsolation(reco::Muon& muon,reco::MuonRef& muonRef ) {
  reco::MuonPFIsolation isoR3;
  if(chargedParticle03_.isValid()) {
       isoR3.sumChargedParticlePt = (*chargedParticle03_)[muonRef];
  }
  else {    return -1;}

  if(chargedHadron03_.isValid()) {
       isoR3.sumChargedHadronPt = (*chargedHadron03_)[muonRef];
  }
  else {    return -1;}

  if(neutralHadron03_.isValid()) {
       isoR3.sumNeutralHadronEt = (*neutralHadron03_)[muonRef];
  }
  else {    return -1;}

  if(photon03_.isValid()) {
       isoR3.sumPhotonEt = (*photon03_)[muonRef];
  }
  else {    return -1;}

  if(pu03_.isValid()) {
       isoR3.sumPUPt = (*pu03_)[muonRef];
  }
  else {    return -1;}



  reco::MuonPFIsolation isoR4;
  if(chargedParticle04_.isValid()) {
       isoR4.sumChargedParticlePt = (*chargedParticle04_)[muonRef];
  }
  else {    return -1;}

  if(chargedHadron04_.isValid()) {
       isoR4.sumChargedHadronPt = (*chargedHadron04_)[muonRef];
  }
  else {    return -1;}

  if(neutralHadron04_.isValid()) {
       isoR4.sumNeutralHadronEt = (*neutralHadron04_)[muonRef];
  }
  else {    return -1;}

  if(photon04_.isValid()) {
       isoR4.sumPhotonEt = (*photon04_)[muonRef];
  }
  else {    return -1;}

  if(pu04_.isValid()) {
       isoR4.sumPUPt = (*pu04_)[muonRef];
  }
  else {    return -1;}


  muon.setPFIsolation(isoR3,isoR4);
					
  return 0;
}



void MuPFIsoHelper::beginEvent(const edm::Event& iEvent){

  iEvent.getByLabel(isoCfg03_.getParameter<edm::InputTag>("chargedParticle"),chargedParticle03_);
  iEvent.getByLabel(isoCfg03_.getParameter<edm::InputTag>("chargedHadron"),chargedHadron03_);
  iEvent.getByLabel(isoCfg03_.getParameter<edm::InputTag>("neutralHadron"),neutralHadron03_);
  iEvent.getByLabel(isoCfg03_.getParameter<edm::InputTag>("photon"),photon03_);
  iEvent.getByLabel(isoCfg03_.getParameter<edm::InputTag>("pu"),pu03_);

  iEvent.getByLabel(isoCfg04_.getParameter<edm::InputTag>("chargedParticle"),chargedParticle04_);
  iEvent.getByLabel(isoCfg04_.getParameter<edm::InputTag>("chargedHadron"),chargedHadron04_);
  iEvent.getByLabel(isoCfg04_.getParameter<edm::InputTag>("neutralHadron"),neutralHadron04_);
  iEvent.getByLabel(isoCfg04_.getParameter<edm::InputTag>("photon"),photon04_);
  iEvent.getByLabel(isoCfg04_.getParameter<edm::InputTag>("pu"),pu04_);

}

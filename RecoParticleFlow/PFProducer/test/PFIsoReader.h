#ifndef RECOPARTICLEFLOW_PFISOREADER_H
#define RECOPARTICLEFLOW_PFISOREADER_H
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include <iostream>
#include <string>
#include <map>

class PFIsoReader : public edm::one::EDAnalyzer<> {
public:
  explicit PFIsoReader(const edm::ParameterSet&);
  ~PFIsoReader() override = default;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& c) override;

private:
  typedef std::vector<edm::Handle<edm::ValueMap<reco::IsoDeposit> > > IsoDepositMaps;
  void printIsoDeposits(const IsoDepositMaps& depmap, const reco::PFCandidatePtr& ptr) const;

  const edm::InputTag inputTagGsfElectrons_;
  const edm::InputTag inputTagPhotons_;
  const edm::InputTag inputTagPFCandidates_;
  const edm::InputTag inputTagValueMapPhotons_;
  const edm::InputTag inputTagValueMapElectrons_;
  const edm::InputTag inputTagValueMapMerged_;
  const std::vector<edm::InputTag> inputTagElectronIsoDeposits_;
  const std::vector<edm::InputTag> inputTagPhotonIsoDeposits_;
  const bool useValueMaps_;

  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> elecToken_;
  const edm::EDGetTokenT<reco::PhotonCollection> photonToken_;
  const edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr> > elecMapToken_;
  const edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr> > photonMapToken_;
  const edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr> > mergeMapToken_;
  std::vector<edm::EDGetTokenT<edm::Handle<edm::ValueMap<reco::IsoDeposit> > > > isoElecToken_;
  std::vector<edm::EDGetTokenT<edm::Handle<edm::ValueMap<reco::IsoDeposit> > > > isoPhotToken_;
};
#endif

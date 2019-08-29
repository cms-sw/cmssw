// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTRecoEcalCandidateProducers
//
/**\class EgammaHLTRecoEcalCandidateProducers.h EgammaHLTRecoEcalCandidateProducers.cc  RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRecoEcalCandidateProducers.h.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id:
//
//

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>
#include <vector>
#include <memory>

class EgammaHLTRecoEcalCandidateProducers : public edm::global::EDProducer<> {
public:
  EgammaHLTRecoEcalCandidateProducers(const edm::ParameterSet& ps);
  ~EgammaHLTRecoEcalCandidateProducers() override;

  void produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::SuperClusterCollection> scHybridBarrelProducer_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> scIslandEndcapProducer_;
  const std::string recoEcalCandidateCollection_;
};

EgammaHLTRecoEcalCandidateProducers::EgammaHLTRecoEcalCandidateProducers(const edm::ParameterSet& config)
    : scHybridBarrelProducer_(
          consumes<reco::SuperClusterCollection>(config.getParameter<edm::InputTag>("scHybridBarrelProducer"))),
      scIslandEndcapProducer_(
          consumes<reco::SuperClusterCollection>(config.getParameter<edm::InputTag>("scIslandEndcapProducer"))),
      recoEcalCandidateCollection_(config.getParameter<std::string>("recoEcalCandidateCollection")) {
  // Register the product
  produces<reco::RecoEcalCandidateCollection>(recoEcalCandidateCollection_);
}

EgammaHLTRecoEcalCandidateProducers::~EgammaHLTRecoEcalCandidateProducers() {}

void EgammaHLTRecoEcalCandidateProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("scHybridBarrelProducer"), edm::InputTag("correctedHybridSuperClusters"));
  desc.add<edm::InputTag>(("scIslandEndcapProducer"), edm::InputTag("correctedEndcapSuperClustersWithPreshower"));
  desc.add<std::string>(("recoEcalCandidateCollection"), "");
  descriptions.add(("hltEgammaHLTRecoEcalCandidateProducers"), desc);
}

void EgammaHLTRecoEcalCandidateProducers::produce(edm::StreamID sid,
                                                  edm::Event& theEvent,
                                                  const edm::EventSetup& theEventSetup) const {
  using namespace edm;

  //
  // create empty output collections
  //

  reco::RecoEcalCandidateCollection outputRecoEcalCandidateCollection;
  auto outputRecoEcalCandidateCollection_p = std::make_unique<reco::RecoEcalCandidateCollection>();

  // Get the  Barrel Super Cluster collection
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByToken(scHybridBarrelProducer_, scBarrelHandle);
  // Get the  Endcap Super Cluster collection
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByToken(scIslandEndcapProducer_, scEndcapHandle);

  //  Loop over barrel SC and fill the  recoecal collection
  int iSC = 0;  // index in recoecal collection
  int lSC = 0;  // local index on barrel

  for (reco::SuperClusterCollection::const_iterator aClus = scBarrelHandle->begin(); aClus != scBarrelHandle->end();
       aClus++) {
    const reco::Particle::Point vtx(0, 0, 0);

    // compute correctly the momentum vector of the recoecal from primary vertex and cluster position
    math::XYZVector direction = aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy());

    reco::RecoEcalCandidate newCandidate(0, p4, vtx);

    outputRecoEcalCandidateCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
    outputRecoEcalCandidateCollection[iSC].setSuperCluster(scRef);

    lSC++;
    iSC++;
  }

  //  Loop over Endcap SC and fill the  recoecal collection
  lSC = 0;  // reset local index for endcap

  for (reco::SuperClusterCollection::const_iterator aClus = scEndcapHandle->begin(); aClus != scEndcapHandle->end();
       aClus++) {
    const reco::Particle::Point vtx(0, 0, 0);

    math::XYZVector direction = aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy());

    reco::RecoEcalCandidate newCandidate(0, p4, vtx);

    outputRecoEcalCandidateCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
    outputRecoEcalCandidateCollection[iSC].setSuperCluster(scRef);

    iSC++;
    lSC++;
  }

  // put the product in the event
  outputRecoEcalCandidateCollection_p->assign(outputRecoEcalCandidateCollection.begin(),
                                              outputRecoEcalCandidateCollection.end());
  theEvent.put(std::move(outputRecoEcalCandidateCollection_p), recoEcalCandidateCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTRecoEcalCandidateProducers);

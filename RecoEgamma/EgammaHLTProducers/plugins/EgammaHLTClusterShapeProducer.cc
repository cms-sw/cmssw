// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTClusterShapeProducer
//
/**\class EgammaHLTClusterShapeProducer EgammaHLTClusterShapeProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTClusterShapeProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTClusterShapeProducer.h,v 1.1 2009/01/15 14:28:27 covarell Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EgammaHLTClusterShapeProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTClusterShapeProducer(const edm::ParameterSet&);
  ~EgammaHLTClusterShapeProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEBToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEEToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterLazyToolsESGetTokens_;
  const bool EtaOrIeta_;
};

EgammaHLTClusterShapeProducer::EgammaHLTClusterShapeProducer(const edm::ParameterSet& config)
    : recoEcalCandidateProducer_(consumes(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      ecalRechitEBToken_(consumes(config.getParameter<edm::InputTag>("ecalRechitEB"))),
      ecalRechitEEToken_(consumes(config.getParameter<edm::InputTag>("ecalRechitEE"))),
      ecalClusterLazyToolsESGetTokens_{consumesCollector()},
      EtaOrIeta_(config.getParameter<bool>("isIeta")) {
  //register your products
  produces<reco::RecoEcalCandidateIsolationMap>();
  produces<reco::RecoEcalCandidateIsolationMap>("sigmaIEtaIEta5x5");
}

EgammaHLTClusterShapeProducer::~EgammaHLTClusterShapeProducer() {}

void EgammaHLTClusterShapeProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("ecalRechitEB"), edm::InputTag("hltEcalRegionalEgammaRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>(("ecalRechitEE"), edm::InputTag("hltEcalRegionalEgammaRecHit", "EcalRecHitsEE"));
  desc.add<bool>(("isIeta"), true);
  descriptions.add(("hltEgammaHLTClusterShapeProducer"), desc);
}

void EgammaHLTClusterShapeProducer::produce(edm::StreamID sid,
                                            edm::Event& iEvent,
                                            const edm::EventSetup& iSetup) const {
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_, recoecalcandHandle);

  auto const& ecalClusterLazyToolsESData = ecalClusterLazyToolsESGetTokens_.get(iSetup);

  EcalClusterLazyTools lazyTools(iEvent, ecalClusterLazyToolsESData, ecalRechitEBToken_, ecalRechitEEToken_);
  noZS::EcalClusterLazyTools lazyTools5x5(iEvent, ecalClusterLazyToolsESData, ecalRechitEBToken_, ecalRechitEEToken_);

  reco::RecoEcalCandidateIsolationMap clshMap(recoecalcandHandle);
  reco::RecoEcalCandidateIsolationMap clsh5x5Map(recoecalcandHandle);

  for (unsigned int iRecoEcalCand = 0; iRecoEcalCand < recoecalcandHandle->size(); iRecoEcalCand++) {
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);
    if (recoecalcandref->superCluster()->seed()->seed().det() != DetId::Ecal) {  //HGCAL, skip for now
      clshMap.insert(recoecalcandref, 0);
      clsh5x5Map.insert(recoecalcandref, 0);
      continue;
    }

    std::vector<float> vCov;
    double sigmaee;
    if (EtaOrIeta_) {
      vCov = lazyTools.localCovariances(*(recoecalcandref->superCluster()->seed()));
      sigmaee = sqrt(vCov[0]);
    } else {
      vCov = lazyTools.covariances(*(recoecalcandref->superCluster()->seed()));
      sigmaee = sqrt(vCov[0]);
      double EtaSC = recoecalcandref->eta();
      if (EtaSC > 1.479)
        sigmaee = sigmaee - 0.02 * (EtaSC - 2.3);
    }

    double sigmaee5x5 = sqrt(lazyTools5x5.localCovariances(*(recoecalcandref->superCluster()->seed()))[0]);
    clshMap.insert(recoecalcandref, sigmaee);
    clsh5x5Map.insert(recoecalcandref, sigmaee5x5);
  }

  iEvent.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(clshMap));
  iEvent.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(clsh5x5Map), "sigmaIEtaIEta5x5");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTClusterShapeProducer);

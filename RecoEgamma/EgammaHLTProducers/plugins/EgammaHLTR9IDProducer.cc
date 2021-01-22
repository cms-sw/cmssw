// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTR9IDProducer
//
/**\class EgammaHLTR9IDProducer EgammaHLTR9IDProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9IDProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTR9Producer.h,v 1.2 2010/06/10 16:19:31 ghezzi Exp $
//         modified by Chris Tully (Princeton)
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

// Framework
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

class EgammaHLTR9IDProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTR9IDProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEBToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEEToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;
};

EgammaHLTR9IDProducer::EgammaHLTR9IDProducer(const edm::ParameterSet& config)
    : recoEcalCandidateProducer_(consumes(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      ecalRechitEBToken_(consumes(config.getParameter<edm::InputTag>("ecalRechitEB"))),
      ecalRechitEEToken_(consumes(config.getParameter<edm::InputTag>("ecalRechitEE"))),
      ecalClusterToolsESGetTokens_{consumesCollector()} {
  //register your products
  produces<reco::RecoEcalCandidateIsolationMap>();
  produces<reco::RecoEcalCandidateIsolationMap>("r95x5");
}

void EgammaHLTR9IDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltRecoEcalCandidate"));
  desc.add<edm::InputTag>(("ecalRechitEB"), edm::InputTag("hltEcalRegionalEgammaRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>(("ecalRechitEE"), edm::InputTag("hltEcalRegionalEgammaRecHit", "EcalRecHitsEE"));
  descriptions.add(("hltEgammaHLTR9IDProducer"), desc);
}

// ------------ method called to produce the data  ------------
void EgammaHLTR9IDProducer::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_, recoecalcandHandle);

  auto const& ecalClusterToolsESData = ecalClusterToolsESGetTokens_.get(iSetup);
  EcalClusterLazyTools lazyTools(iEvent, ecalClusterToolsESData, ecalRechitEBToken_, ecalRechitEEToken_);
  noZS::EcalClusterLazyTools lazyTools5x5(iEvent, ecalClusterToolsESData, ecalRechitEBToken_, ecalRechitEEToken_);
  reco::RecoEcalCandidateIsolationMap r9Map(recoecalcandHandle);
  reco::RecoEcalCandidateIsolationMap r95x5Map(recoecalcandHandle);
  for (unsigned int iRecoEcalCand = 0; iRecoEcalCand < recoecalcandHandle->size(); iRecoEcalCand++) {
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);  //-recoecalcandHandle->begin());

    float r9 = -1;
    float r95x5 = -1;

    float e9 = lazyTools.e3x3(*(recoecalcandref->superCluster()->seed()));
    float e95x5 = lazyTools5x5.e3x3(*(recoecalcandref->superCluster()->seed()));

    float eraw = recoecalcandref->superCluster()->rawEnergy();
    if (eraw > 0.) {
      r9 = e9 / eraw;
      r95x5 = e95x5 / eraw;
    }

    r9Map.insert(recoecalcandref, r9);
    r95x5Map.insert(recoecalcandref, r95x5);
  }

  iEvent.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(r9Map));

  iEvent.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(r95x5Map), "r95x5");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTR9IDProducer);

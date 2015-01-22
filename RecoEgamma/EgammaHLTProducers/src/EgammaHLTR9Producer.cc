/** \class EgammaHLTR9Producer
 *
 *  \author Roberto Covarelli (CERN)
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9Producer.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTR9Producer::EgammaHLTR9Producer(const edm::ParameterSet& config):
  recoEcalCandidateProducer_(consumes<reco::RecoEcalCandidateCollection> (config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
  ecalRechitEBToken_(consumes<EcalRecHitCollection>(config.getParameter< edm::InputTag > ("ecalRechitEB"))),
  ecalRechitEEToken_(consumes<EcalRecHitCollection>(config.getParameter< edm::InputTag > ("ecalRechitEE"))),
  useSwissCross_(config.getParameter< bool > ("useSwissCross")) {

  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}

EgammaHLTR9Producer::~EgammaHLTR9Producer()
{}

// ------------ method called to produce the data  ------------

void EgammaHLTR9Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltRecoEcalCandidate"));
  desc.add<edm::InputTag>(("ecalRechitEB"), edm::InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>(("ecalRechitEE"), edm::InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEE"));
  desc.add<bool> (("useSwissCross"), false);
  descriptions.add(("hltEgammaHLTR9Producer"), desc);  
}


void EgammaHLTR9Producer::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);

  EcalClusterLazyTools lazyTools(iEvent, iSetup, ecalRechitEBToken_, ecalRechitEEToken_);
  
  reco::RecoEcalCandidateIsolationMap r9Map;
   
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle,iRecoEcalCand-recoecalcandHandle->begin());

    float r9 = -1;

    if (useSwissCross_){
      edm::Handle< EcalRecHitCollection > pEBRecHits;
      iEvent.getByToken(ecalRechitEBToken_, pEBRecHits);
      r9 = -1;
    }
    else{
    float e9 = lazyTools.e3x3( *(recoecalcandref->superCluster()->seed()) );
    if (e9 != 0 ) {r9 = lazyTools.eMax(*(recoecalcandref->superCluster()->seed())  )/e9;}
    }

    r9Map.insert(recoecalcandref, r9);
    
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> R9Map(new reco::RecoEcalCandidateIsolationMap(r9Map));
  iEvent.put(R9Map);

}

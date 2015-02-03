/** \class EgammaHLTR9IDProducer
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9IDProducer.h"

// Framework
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

EgammaHLTR9IDProducer::EgammaHLTR9IDProducer(const edm::ParameterSet& config):
  recoEcalCandidateProducer_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
  ecalRechitEBToken_(consumes<EcalRecHitCollection>(config.getParameter< edm::InputTag > ("ecalRechitEB"))),
  ecalRechitEEToken_(consumes<EcalRecHitCollection>(config.getParameter< edm::InputTag > ("ecalRechitEE"))) {
  
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
  produces < reco::RecoEcalCandidateIsolationMap >("r95x5");
}

EgammaHLTR9IDProducer::~EgammaHLTR9IDProducer()
{}

void EgammaHLTR9IDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltRecoEcalCandidate"));
  desc.add<edm::InputTag>(("ecalRechitEB"), edm::InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>(("ecalRechitEE"), edm::InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEE"));
  descriptions.add(("hltEgammaHLTR9IDProducer"), desc);  
}


// ------------ method called to produce the data  ------------
void EgammaHLTR9IDProducer::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_, recoecalcandHandle);

  EcalClusterLazyTools lazyTools( iEvent, iSetup, ecalRechitEBToken_, ecalRechitEEToken_ );
  noZS::EcalClusterLazyTools lazyTools5x5(iEvent, iSetup, ecalRechitEBToken_, ecalRechitEEToken_ );
  reco::RecoEcalCandidateIsolationMap r9Map(recoecalcandHandle);
  reco::RecoEcalCandidateIsolationMap r95x5Map(recoecalcandHandle); 
  for(unsigned  int iRecoEcalCand=0; iRecoEcalCand<recoecalcandHandle->size(); iRecoEcalCand++) {
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);//-recoecalcandHandle->begin());

    float r9 = -1;
    float r95x5 = -1;

    float e9 = lazyTools.e3x3( *(recoecalcandref->superCluster()->seed()) );
    float e95x5 = lazyTools5x5.e3x3( *(recoecalcandref->superCluster()->seed()) );

    float eraw = recoecalcandref->superCluster()->rawEnergy();
    if (eraw > 0. ) {
      r9 = e9/eraw;
      r95x5 = e95x5/eraw;
    }

    r9Map.insert(recoecalcandref, r9);
    r95x5Map.insert(recoecalcandref,r95x5);
    
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> R9Map(new reco::RecoEcalCandidateIsolationMap(r9Map));
  iEvent.put(R9Map);

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> R95x5Map(new reco::RecoEcalCandidateIsolationMap(r95x5Map));
  iEvent.put(R95x5Map,"r95x5");
}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTR9IDProducer);

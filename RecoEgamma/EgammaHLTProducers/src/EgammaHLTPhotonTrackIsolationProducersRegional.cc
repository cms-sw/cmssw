/** \class EgammaHLTPhotonTrackIsolationProducersRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 * 
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPhotonTrackIsolationProducersRegional.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTPhotonTrackIsolationProducersRegional::EgammaHLTPhotonTrackIsolationProducersRegional(const edm::ParameterSet& config):
  recoEcalCandidateProducer_(consumes<reco::RecoEcalCandidateCollection> (config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
  trackProducer_            (consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("trackProducer"))),
  countTracks_              (config.getParameter<bool>("countTracks")),
  egTrkIsoPtMin_            (config.getParameter<double>("egTrkIsoPtMin")),
  egTrkIsoConeSize_         (config.getParameter<double>("egTrkIsoConeSize")),
  egTrkIsoZSpan_            (config.getParameter<double>("egTrkIsoZSpan")),
  egTrkIsoRSpan_            (config.getParameter<double>("egTrkIsoRSpan")),
  egTrkIsoVetoConeSize_     (config.getParameter<double>("egTrkIsoVetoConeSize")),
  egTrkIsoStripBarrel_      (config.getParameter<double>("egTrkIsoStripBarrel")),
  egTrkIsoStripEndcap_      (config.getParameter<double>("egTrkIsoStripEndcap")) {
  
  test_ = new EgammaHLTTrackIsolation(egTrkIsoPtMin_, egTrkIsoConeSize_,
				      egTrkIsoZSpan_, egTrkIsoRSpan_, egTrkIsoVetoConeSize_,
				      egTrkIsoStripBarrel_, egTrkIsoStripEndcap_);

  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}

EgammaHLTPhotonTrackIsolationProducersRegional::~EgammaHLTPhotonTrackIsolationProducersRegional() {
  delete test_;
}

void EgammaHLTPhotonTrackIsolationProducersRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("trackProducer"), edm::InputTag("hltL1SeededEgammaRegionalCTFFinalFitWithMaterial"));
  desc.add<bool>(("countTracks"), false);
  desc.add<double>(("egTrkIsoPtMin"), 1.0);
  desc.add<double>(("egTrkIsoConeSize"), 0.29);
  desc.add<double>(("egTrkIsoZSpan"), 999999.0);
  desc.add<double>(("egTrkIsoRSpan"), 999999.0);
  desc.add<double>(("egTrkIsoVetoConeSize"), 0.06);
  desc.add<double>(("egTrkIsoStripBarrel"), 0.03);
  desc.add<double>(("egTrkIsoStripEndcap"), 0.03);
  descriptions.add(("hltEgammaHLTPhotonTrackIsolationProducersRegional"), desc);  
}
  

// ------------ method called to produce the data  ------------
void
EgammaHLTPhotonTrackIsolationProducersRegional::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);

 // Get the barrel hcal hits
  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByToken(trackProducer_, trackHandle);
  const reco::TrackCollection* trackCollection = trackHandle.product();

  reco::RecoEcalCandidateIsolationMap isoMap;
  
  for(unsigned int iRecoEcalCand=0; iRecoEcalCand<recoecalcandHandle->size(); iRecoEcalCand++) {
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);

    bool usePhotonVertex = false;
     
    float isol;
    if (countTracks_) {
      isol = test_->photonTrackCount(&(*recoecalcandref), trackCollection, usePhotonVertex);
    } else {
      isol = test_->photonPtSum(&(*recoecalcandref), trackCollection, usePhotonVertex);
    }
    
    isoMap.insert(recoecalcandref, isol);

  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}

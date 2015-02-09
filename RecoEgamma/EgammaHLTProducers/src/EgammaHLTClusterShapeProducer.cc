/** \class EgammaHLTClusterShapeProducer
 *
 *  \author Roberto Covarelli (CERN)
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTClusterShapeProducer.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTClusterShapeProducer::EgammaHLTClusterShapeProducer(const edm::ParameterSet& config) : 
  recoEcalCandidateProducer_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
  ecalRechitEBToken_(consumes<EcalRecHitCollection>(config.getParameter< edm::InputTag > ("ecalRechitEB"))),
  ecalRechitEEToken_(consumes<EcalRecHitCollection>(config.getParameter< edm::InputTag > ("ecalRechitEE"))),
  EtaOrIeta_(config.getParameter< bool > ("isIeta")) {
   
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
  produces < reco::RecoEcalCandidateIsolationMap >("sigmaIEtaIEta5x5");
}

EgammaHLTClusterShapeProducer::~EgammaHLTClusterShapeProducer()
{}

void EgammaHLTClusterShapeProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add< edm::InputTag >(("ecalRechitEB"), edm::InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB"));
  desc.add< edm::InputTag >(("ecalRechitEE"), edm::InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEE"));
  desc.add< bool >(("isIeta"), true);
  descriptions.add(("hltEgammaHLTClusterShapeProducer"), desc);  
}

void EgammaHLTClusterShapeProducer::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);

  EcalClusterLazyTools lazyTools( iEvent, iSetup, ecalRechitEBToken_, ecalRechitEEToken_ );
  noZS::EcalClusterLazyTools lazyTools5x5(iEvent, iSetup, ecalRechitEBToken_, ecalRechitEEToken_ );

  reco::RecoEcalCandidateIsolationMap clshMap;
  reco::RecoEcalCandidateIsolationMap clsh5x5Map;
  
 
  for(unsigned int iRecoEcalCand = 0; iRecoEcalCand<recoecalcandHandle->size(); iRecoEcalCand++) {
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);
    
    std::vector<float> vCov ; 
    double sigmaee;
    if (EtaOrIeta_) {
      vCov = lazyTools.localCovariances( *(recoecalcandref->superCluster()->seed()) );
      sigmaee = sqrt(vCov[0]);
    } else {
      vCov = lazyTools.covariances( *(recoecalcandref->superCluster()->seed()) );
      sigmaee = sqrt(vCov[0]);
      double EtaSC = recoecalcandref->eta();
      if (EtaSC > 1.479) sigmaee = sigmaee - 0.02*(EtaSC - 2.3); 
    }
    
    double sigmaee5x5 = sqrt(lazyTools5x5.localCovariances(*(recoecalcandref->superCluster()->seed()) )[0]);
    clshMap.insert(recoecalcandref, sigmaee);
    clsh5x5Map.insert(recoecalcandref,sigmaee5x5);

  
  }

  

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> clushMap(new reco::RecoEcalCandidateIsolationMap(clshMap));
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> clush5x5Map(new reco::RecoEcalCandidateIsolationMap(clsh5x5Map));
  iEvent.put(clushMap);
  iEvent.put(clush5x5Map,"sigmaIEtaIEta5x5");
}

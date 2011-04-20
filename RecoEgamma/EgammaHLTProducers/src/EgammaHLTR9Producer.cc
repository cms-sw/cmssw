/** \class EgammaHLTR9Producer
 *
 *  \author Roberto Covarelli (CERN)
 *
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9Producer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

EgammaHLTR9Producer::EgammaHLTR9Producer(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_ = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");
  ecalRechitEBTag_ = conf_.getParameter< edm::InputTag > ("ecalRechitEB");
  ecalRechitEETag_ = conf_.getParameter< edm::InputTag > ("ecalRechitEE");
  useSwissCross_   = conf_.getParameter< bool > ("useSwissCross");
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}


EgammaHLTR9Producer::~EgammaHLTR9Producer(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTR9Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);

  EcalClusterLazyTools lazyTools( iEvent, iSetup, ecalRechitEBTag_, ecalRechitEETag_ );
  
  reco::RecoEcalCandidateIsolationMap r9Map;
   
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle,iRecoEcalCand-recoecalcandHandle->begin());

    float r9 = -1;

    if (useSwissCross_){
      DetId maxEId = (lazyTools.getMaximum(*(recoecalcandref->superCluster()->seed()) )).first;
      //float EcalSeverityLevelAlgo::swissCross( const DetId id, const EcalRecHitCollection & recHits, float recHitEtThreshold )
      edm::Handle< EcalRecHitCollection > pEBRecHits;
      iEvent.getByLabel( ecalRechitEBTag_, pEBRecHits );
      r9 = EcalSeverityLevelAlgo::swissCross( maxEId, *(pEBRecHits.product()), 0. );
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

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTR9Producer);

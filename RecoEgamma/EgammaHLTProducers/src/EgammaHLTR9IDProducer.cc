/** \class EgammaHLTR9IDProducer
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9IDProducer.h"

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

EgammaHLTR9IDProducer::EgammaHLTR9IDProducer(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
  recoEcalCandidateProducer_ = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");
  ecalRechitEBTag_ = conf_.getParameter< edm::InputTag > ("ecalRechitEB");
  ecalRechitEETag_ = conf_.getParameter< edm::InputTag > ("ecalRechitEE");
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}


EgammaHLTR9IDProducer::~EgammaHLTR9IDProducer(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTR9IDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);

  EcalClusterLazyTools lazyTools( iEvent, iSetup, ecalRechitEBTag_, ecalRechitEETag_ );
  
  reco::RecoEcalCandidateIsolationMap r9Map;
   
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){
    
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle,iRecoEcalCand-recoecalcandHandle->begin());

    float r9 = -1;

    float e9 = lazyTools.e3x3( *(recoecalcandref->superCluster()->seed()) );
    float eraw = recoecalcandref->superCluster()->rawEnergy();
    if (eraw > 0. ) {r9 = e9/eraw;}

    r9Map.insert(recoecalcandref, r9);
    
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> R9Map(new reco::RecoEcalCandidateIsolationMap(r9Map));
  iEvent.put(R9Map);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTR9IDProducer);

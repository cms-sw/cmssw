/** \class EgammaHLTPhotonTrackIsolationProducers
 *
 *  \author Monica Vazquez Acosta (CERN)
 * 
 * $Id:
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPhotonTrackIsolationProducers.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"




EgammaHLTPhotonTrackIsolationProducers::EgammaHLTPhotonTrackIsolationProducers(const edm::ParameterSet& config) : conf_(config)
{

  recoEcalCandidateProducer_    = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");
  trackProducer_                = conf_.getParameter<edm::InputTag>("trackProducer");

  egTrkIsoPtMin_                = conf_.getParameter<double>("egTrkIsoPtMin");
  egTrkIsoConeSize_             = conf_.getParameter<double>("egTrkIsoConeSize");
  egTrkIsoZSpan_                = conf_.getParameter<double>("egTrkIsoZSpan");
  egTrkIsoRSpan_                = conf_.getParameter<double>("egTrkIsoRSpan");
  egTrkIsoVetoConeSize_         = conf_.getParameter<double>("egTrkIsoVetoConeSize");


  test_ = new EgammaHLTTrackIsolation(egTrkIsoPtMin_,egTrkIsoConeSize_,
				      egTrkIsoZSpan_,egTrkIsoRSpan_,egTrkIsoVetoConeSize_);


  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();

}


EgammaHLTPhotonTrackIsolationProducers::~EgammaHLTPhotonTrackIsolationProducers(){delete test_;}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTPhotonTrackIsolationProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get the HLT filtered objects
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);

 // Get the barrel hcal hits
  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByLabel(trackProducer_, trackHandle);
  const reco::TrackCollection* trackCollection = trackHandle.product();


  reco::RecoEcalCandidateIsolationMap isoMap;


  for(reco::HLTFilterObjectWithRefs::const_iterator iRecoEcalCand = recoecalcandHandle->begin(); iRecoEcalCand != recoecalcandHandle->end(); iRecoEcalCand++){

    
    const reco::RecoEcalCandidateRef recoecalcandref(reco::RecoEcalCandidateRef((recoecalcandHandle->getParticleRef(iRecoEcalCand-recoecalcandHandle->begin())).castTo<reco::RecoEcalCandidateRef>()));

    bool usePhotonVertex = false;

    const reco::RecoCandidate *tempiRecoEcalCand = &(*recoecalcandref);
    float isol =  test_->photonTrackCount(tempiRecoEcalCand,trackCollection,usePhotonVertex);

    isoMap.insert(recoecalcandref, isol);

  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTTrackIsolationProducers);

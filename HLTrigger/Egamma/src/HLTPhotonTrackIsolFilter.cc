/** \class HLTPhotonTrackIsolFilter
 *
 * $Id: HLTPhotonTrackIsolFilter.cc,v 1.3 2007/03/07 10:44:05 monicava Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTPhotonTrackIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTPhotonTrackIsolFilter::HLTPhotonTrackIsolFilter(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  numtrackisolcut_  = iConfig.getParameter<double> ("numtrackisolcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTPhotonTrackIsolFilter::~HLTPhotonTrackIsolFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTPhotonTrackIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  //get hold of track isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of track isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
  // look at all photons,  check cuts and add to filter object
  int n = 0;
  
  edm::RefToBase<reco::Candidate> candref;
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    candref = recoecalcands->getParticleRef(i);
    
    reco::RecoEcalCandidateRef phr = candref.castTo<reco::RecoEcalCandidateRef>();
    
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( phr );
    
    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( phr ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }

    float vali = mapi->val;
    
    if ( vali < numtrackisolcut_) {
      n++;
      filterproduct->putParticle(candref);
    }
  }
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);

   return accept;
}


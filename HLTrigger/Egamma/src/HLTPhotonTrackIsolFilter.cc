/** \class HLTPhotonTrackIsolFilter
 *
 * $Id: HLTPhotonTrackIsolFilter.cc,v 1.4 2007/04/02 17:14:14 mpieri Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTPhotonTrackIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

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
produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTPhotonTrackIsolFilter::~HLTPhotonTrackIsolFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTPhotonTrackIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerPhoton, recoecalcands);


  
  //get hold of track isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of track isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
  // look at all photons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    
    ref = recoecalcands[i];
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );
    
    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( ref ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }

    float vali = mapi->val;
    
    if ( vali < numtrackisolcut_) {
      n++;
      filterproduct->addObject(TriggerPhoton, ref);
    }
  }
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);

   return accept;
}


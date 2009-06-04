/** \class HLTPhotonTrackIsolFilter
 *
 * $Id: HLTPhotonTrackIsolFilter.cc,v 1.9 2009/01/20 11:30:38 covarell Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTPhotonTrackIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
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
  ptOrNumtrackisolcut_  = iConfig.getParameter<double> ("ptOrNumtrackisolcut");
  pttrackisolOverEcut_ = iConfig.getParameter<double> ("pttrackisolOverEcut");
  pttrackisolOverE2cut_ = iConfig.getParameter<double> ("pttrackisolOverE2cut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

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
  if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);


  
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
    float energy = ref->superCluster()->energy();

    if ( vali <= ptOrNumtrackisolcut_) {
      n++;
      filterproduct->addObject(TriggerPhoton, ref);
      continue;
    }
    if (energy > 0.)
      if ( vali/energy <= pttrackisolOverEcut_ || vali/(energy*energy) <= pttrackisolOverE2cut_ ) {
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


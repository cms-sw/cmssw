/** \class EgammaHLTEcalIsolFilter
 *
 * $Id: HLTEgammaEcalIsolFilter.cc,v 1.3 2007/03/07 10:44:05 monicava Exp $
 * 
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaEcalIsolFilter.h"

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
HLTEgammaEcalIsolFilter::HLTEgammaEcalIsolFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  ecalisolcut_  = iConfig.getParameter<double> ("ecalisolcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaEcalIsolFilter::~HLTEgammaEcalIsolFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaEcalIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> candref;
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter i "<<i<<" "<<std::endl;
    candref = recoecalcands->getParticleRef(i);
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter 1 "<<std::endl;
    
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter 11 "<<std::endl;
    
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter recr "<<recr<<" "<<std::endl;
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( recr );
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter 12 "<<std::endl;
    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( recr ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }
    //if(!mapi)  
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter 1000 "<<std::endl;
    float vali = mapi->val;
    //std::cout<<"MARCO HLTEgammaEcalIsolFilter vali "<<vali<<" "<<std::endl;
    
    if ( vali < ecalisolcut_) {
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


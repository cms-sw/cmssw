/** \class EgammaHLTEcalIsolFilter
 *
 * $Id: HLTEgammaEcalIsolFilter.cc,v 1.2 2007/01/26 18:40:21 monicava Exp $
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
  ecalisolcut_  = iConfig.getParameter<double> ("ecalisolcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");

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
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    candref = recoecalcands->getParticleRef(i);
    
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( recr );
    
    float vali = mapi->val;
    
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


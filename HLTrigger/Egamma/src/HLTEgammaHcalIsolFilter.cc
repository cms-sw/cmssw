/** \class HLTEgammaHcalIsolFilter
 *
 * $Id: HLTEgammaHcalIsolFilter.cc,v 1.4 2007/03/07 19:13:59 monicava Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaHcalIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

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
HLTEgammaHcalIsolFilter::HLTEgammaHcalIsolFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  hcalisolbarrelcut_  = iConfig.getParameter<double> ("hcalisolbarrelcut");
  hcalisolendcapcut_  = iConfig.getParameter<double> ("hcalisolendcapcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaHcalIsolFilter::~HLTEgammaHcalIsolFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaHcalIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> candref;
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  //get hold of hcal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of hcal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);

  // look at all photons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    //std::cout<<"MARCO HLTEgammaHcalIsolFilter i "<<i<<" "<<std::endl;
    candref = recoecalcands->getParticleRef(i);
    //std::cout<<"MARCO HLTEgammaHcalIsolFilter candref "<<(long) candref<<" "<<std::endl;    
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    //std::cout<<"MARCO HLTEgammaHcalIsolFilter recr "<<recr<<" "<<std::endl;
    
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( recr );
    
    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( recr ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }
     float vali = mapi->val;
     //std::cout<<"MARCO HLTEgammaHcalIsolFilter vali "<<vali<<" "<<std::endl;
     
     if(fabs(recoecalcands->getParticleRef(i).get()->eta()) < 1.5){
       if ( vali < hcalisolbarrelcut_) {
	 n++;
	 filterproduct->putParticle(candref);
       }
     }
     if(
	(fabs(recoecalcands->getParticleRef(i).get()->eta()) > 1.5) && 
	(fabs(recoecalcands->getParticleRef(i).get()->eta()) < 2.5)
	){
       if ( vali < hcalisolendcapcut_) {
	 n++;
	 filterproduct->putParticle(candref);
       }
     }
  }
  
   // filter decision
   bool accept(n>=ncandcut_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   return accept;
}


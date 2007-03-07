/** \class HLTEgammaHcalIsolFilter
 *
 * $Id: HLTEgammaHcalIsolFilter.cc,v 1.3 2007/03/07 10:44:05 monicava Exp $
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
  hcalisolbarrelcut_  = iConfig.getParameter<double> ("hcalisolbarrelcut");
  hcalisolendcapcut_  = iConfig.getParameter<double> ("hcalisolendcapcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");

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
  
  // look at all photons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    candref = recoecalcands->getParticleRef(i);
    
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( recr );
    
     float vali = mapi->val;
     
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


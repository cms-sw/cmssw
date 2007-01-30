/** \class HLTEgammaEtFilter
 *
 * $Id: HLTEgammaEtFilter.cc,v 1.2 2007/01/26 18:40:21 monicava Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaEtFilter.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//
// constructors and destructor
//
HLTEgammaEtFilter::HLTEgammaEtFilter(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
   etcut_  = iConfig.getParameter<double> ("etcut");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaEtFilter::~HLTEgammaEtFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaEtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (inputTag_,recoecalcands);
  
  // look at all candidates,  check cuts and add to filter object
  int n(0);

  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    ref = recoecalcands->getParticleRef(i);

    if ( (recoecalcands->getParticleRef(i).get()->et() ) >= etcut_) {
      n++;
      filterproduct->putParticle(ref);
    }
  }


  /*
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoecalcands->begin(); recoecalcand!=recoecalcands->end(); recoecalcand++) {
    
    if ( (recoecalcand->et()) >= etcut_) {
      n++;
      ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoecalcands,distance(recoecalcands->begin(),recoecalcand)));
      filterproduct->putParticle(ref);
    }
  }
  */
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}

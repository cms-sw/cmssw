/** \class HLTEgammaEtFilter
 *
 * $Id: HLTEgammaEtFilter.cc,v 1.10 2009/01/27 13:57:07 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaEtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//
// constructors and destructor
//
HLTEgammaEtFilter::HLTEgammaEtFilter(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
   etcutEB_  = iConfig.getParameter<double> ("etcutEB");
   etcutEE_  = iConfig.getParameter<double> ("etcutEE");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");
   store_ = iConfig.getParameter<bool>("saveTags") ;
   relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
   L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
   L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaEtFilter::~HLTEgammaEtFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaEtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace trigger;
  // The filter object
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
    if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
    if( store_ && relaxed_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}

  // Ref to Candidate object to be recorded in filter object
   edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // get hold of filtered candidates
  //edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (inputTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;                // vref with your specific C++ collection type
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
 
  // look at all candidates,  check cuts and add to filter object
  int n(0);

  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    
    ref = recoecalcands[i] ;
    
    if( ( fabs(ref->eta()) < 1.479 &&  ref->et()  >= etcutEB_ ) || ( fabs(ref->eta()) >= 1.479 &&  ref->et()  >= etcutEE_ ) ){
      n++;
      // std::cout << "Passed eta: " << ref->eta() << std::endl;
      filterproduct->addObject(TriggerCluster, ref);
    }
  }
  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}

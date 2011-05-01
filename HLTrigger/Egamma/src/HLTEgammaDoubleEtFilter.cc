/** \class HLTEgammaDoubleEtFilter
 *
 * $Id: HLTEgammaDoubleEtFilter.cc,v 1.7 2008/05/06 13:58:06 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

class EgammaHLTEtSortCriterium{
public:
  bool operator() (edm::Ref<reco::RecoEcalCandidateCollection> lhs, edm::Ref<reco::RecoEcalCandidateCollection> rhs){
    return lhs->et() > rhs->et();
  }
};

//
// constructors and destructor
//
HLTEgammaDoubleEtFilter::HLTEgammaDoubleEtFilter(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   etcut1_  = iConfig.getParameter<double> ("etcut1");
   etcut2_  = iConfig.getParameter<double> ("etcut2");
   npaircut_  = iConfig.getParameter<int> ("npaircut");
   store_ = iConfig.getParameter<bool>("saveTags") ;
   relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
   L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
   L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 
   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaDoubleEtFilter::~HLTEgammaDoubleEtFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaDoubleEtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // The filter object
  using namespace trigger;
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
    if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
    if( store_ && relaxed_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  // Ref to Candidate object to be recorded in filter object
   edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> >  mysortedrecoecalcands;
  PrevFilterOutput->getObjects(TriggerPhoton,  mysortedrecoecalcands);

  // look at all candidates,  check cuts and add to filter object
  int n(0);
  
  // Sort the list
  std::sort(mysortedrecoecalcands.begin(), mysortedrecoecalcands.end(), EgammaHLTEtSortCriterium());
  edm::Ref<reco::RecoEcalCandidateCollection> ref1, ref2;
  for (unsigned int i=0; i<mysortedrecoecalcands.size(); i++) {
    ref1 = mysortedrecoecalcands[i];
    if( ref1->et() >= etcut1_){
      for (unsigned int j=i+1; j<mysortedrecoecalcands.size(); j++) {
	ref2 = mysortedrecoecalcands[j];
	if( ref2->et() >= etcut2_ ){
	  filterproduct->addObject(TriggerPhoton, ref1);
	  filterproduct->addObject(TriggerPhoton, ref2);
	  n++;
	}
      }
    }
  }
  

  // filter decision
  bool accept(n>=npaircut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}


  

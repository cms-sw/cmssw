/** \class HLTEgammaDoubleEtFilter
 *
 * $Id: HLTEgammaDoubleEtFilter.cc,v 1.1 2007/01/26 10:37:17 monicava Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtFilter.h"

#include "FWCore/Framework/interface/Handle.h"


#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

class EtSortCriterium{
public:
  bool operator() (edm::RefToBase<reco::Candidate> lhs, edm::RefToBase<reco::Candidate> rhs){
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

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaDoubleEtFilter::~HLTEgammaDoubleEtFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaDoubleEtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands); 
  
  // look at all candidates,  check cuts and add to filter object
  int n(0);
  
  // Create sorted list
  std::vector<edm::RefToBase< reco::Candidate > > mysortedrecoecalcands;
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    mysortedrecoecalcands.push_back(recoecalcands->getParticleRef(i));
  }
  // Sort the vector using predicate and std::sort
  std::sort(mysortedrecoecalcands.begin(), mysortedrecoecalcands.end(), EtSortCriterium());
  
  for (unsigned int i=0; i<mysortedrecoecalcands.size(); i++) {
    edm::RefToBase<reco::Candidate> ref1 = mysortedrecoecalcands[i];
    if( ref1->et() >= etcut1_){
      
      for (unsigned int j=0; j<mysortedrecoecalcands.size(); j++) {
	edm::RefToBase<reco::Candidate> ref2 = mysortedrecoecalcands[j];
	if( ref2->et() >= etcut2_ && (i != j ) && (i < j) ){
	  filterproduct->putParticle(ref1);
	  filterproduct->putParticle(ref2);
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


  

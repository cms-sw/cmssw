/** \class HLTEgammaDoubleEtPhiFilter
 *
 * $Id: HLTEgammaDoubleEtPhiFilter.cc,v 1.3 2007/03/07 10:44:05 monicava Exp $
 *
 *  \author Jonathan Hollar (LLNL)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtPhiFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

class HLTEgammaEtSortCriterium{
public:
  bool operator() (edm::RefToBase<reco::Candidate> lhs, edm::RefToBase<reco::Candidate> rhs){
    return lhs->et() > rhs->et();
  }
};

//
// constructors and destructor
//
HLTEgammaDoubleEtPhiFilter::HLTEgammaDoubleEtPhiFilter(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   etcut1_  = iConfig.getParameter<double> ("etcut1");
   etcut2_  = iConfig.getParameter<double> ("etcut2");
   min_Acop_ =   iConfig.getParameter<double> ("MinAcop");
   max_Acop_ =   iConfig.getParameter<double> ("MaxAcop");
   min_EtBalance_ = iConfig.getParameter<double> ("MinEtBalance");
   max_EtBalance_ = iConfig.getParameter<double> ("MaxEtBalance");
   npaircut_  = iConfig.getParameter<int> ("npaircut");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaDoubleEtPhiFilter::~HLTEgammaDoubleEtPhiFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaDoubleEtPhiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
  std::sort(mysortedrecoecalcands.begin(), mysortedrecoecalcands.end(),  HLTEgammaEtSortCriterium());
  
  for (unsigned int i=0; i<mysortedrecoecalcands.size(); i++) {
    edm::RefToBase<reco::Candidate> ref1 = mysortedrecoecalcands[i];
    if( ref1->et() >= etcut1_){
      
      for (unsigned int j=0; j<mysortedrecoecalcands.size(); j++) {
	edm::RefToBase<reco::Candidate> ref2 = mysortedrecoecalcands[j];
	if( ref2->et() >= etcut2_ && (i != j ) && (i < j) ){
	  
	  // Acoplanarity
	  double acop = fabs(ref1->phi()-ref2->phi());
	  if (acop>M_PI) acop = 2*M_PI - acop;
	  acop = M_PI - acop;
	  LogDebug("HLTEgammaDoubleEtPhiFilter") << " ... 1-2 acop= " << acop;

	  if ((acop>=min_Acop_) && (acop<=max_Acop_))
	  {
            // Et balance
            double etbalance = fabs(ref1->et()-ref2->et());
            if ((etbalance>=min_EtBalance_) && (etbalance<=max_EtBalance_))
	    {
	      filterproduct->putParticle(ref1);
	      filterproduct->putParticle(ref2);
	      n++;
	    }
	  }
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


  

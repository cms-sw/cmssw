/** \class HLTEgammaDoubleEtPhiFilter
 *
 * $Id: HLTEgammaDoubleEtPhiFilter.cc,v 1.6 2012/03/06 10:13:59 sharper Exp $
 *
 *  \author Jonathan Hollar (LLNL)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtPhiFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

class EgammaHLTEtSortCriterium{
public:
  bool operator() (edm::Ref<reco::RecoEcalCandidateCollection> lhs, edm::Ref<reco::RecoEcalCandidateCollection> rhs) {
    return lhs->et() > rhs->et();
  }
};

//
// constructors and destructor
//
HLTEgammaDoubleEtPhiFilter::HLTEgammaDoubleEtPhiFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   etcut1_  = iConfig.getParameter<double> ("etcut1");
   etcut2_  = iConfig.getParameter<double> ("etcut2");
   min_Acop_ =   iConfig.getParameter<double> ("MinAcop");
   max_Acop_ =   iConfig.getParameter<double> ("MaxAcop");
   min_EtBalance_ = iConfig.getParameter<double> ("MinEtBalance");
   max_EtBalance_ = iConfig.getParameter<double> ("MaxEtBalance");
   npaircut_  = iConfig.getParameter<int> ("npaircut");
}

HLTEgammaDoubleEtPhiFilter::~HLTEgammaDoubleEtPhiFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaDoubleEtPhiFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace trigger;

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (candTag_,PrevFilterOutput);
  
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> >  mysortedrecoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster,  mysortedrecoecalcands);
  if(mysortedrecoecalcands.empty()) PrevFilterOutput->getObjects(TriggerCluster,mysortedrecoecalcands);  //we dont know if its type trigger cluster or trigger photon
  // Sort the list
  std::sort(mysortedrecoecalcands.begin(), mysortedrecoecalcands.end(), EgammaHLTEtSortCriterium());
  edm::Ref<reco::RecoEcalCandidateCollection> ref1, ref2;

  int n(0);
  for (unsigned int i=0; i<mysortedrecoecalcands.size(); i++) {
    ref1 = mysortedrecoecalcands[i];
    if( ref1->et() >= etcut1_){
      
      for (unsigned int j=i+1; j<mysortedrecoecalcands.size(); j++) {
	ref2 = mysortedrecoecalcands[j];
	if( ref2->et() >= etcut2_ ){
	  
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
	      filterproduct.addObject(TriggerCluster, ref1);
	      filterproduct.addObject(TriggerCluster, ref2);
	      n++;
	    }
	  }
	}
      }
    }
  }


  // filter decision
  bool accept(n>=npaircut_);
  
  return accept;
}


  

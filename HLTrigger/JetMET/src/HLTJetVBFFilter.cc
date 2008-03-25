/** \class HLTJetVBFFilter
 *
 * $Id: HLTJetVBFFilter.cc,v 1.2 2007/03/08 22:46:25 apana Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTJetVBFFilter::HLTJetVBFFilter(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
   minEt_   = iConfig.getParameter<double> ("minEt");
   minDeltaEta_= iConfig.getParameter<double> ("minDeltaEta"); 

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTJetVBFFilter::~HLTJetVBFFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTJetVBFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  // Get the recoEcalCandidates

  edm::Handle<reco::CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(recocalojets->size() > 1){
    // events with two or more jets

    double etjet1=0.;
    double etjet2=0.;
    double etajet1=0.;
    double etajet2=0.;
    int countjets =0;

    for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet<=(recocalojets->begin()+1); recocalojet++) {
      
      if(countjets==0) {
	etjet1 = recocalojet->et();
	etajet1 = recocalojet->eta();
      }
      if(countjets==1) {
	etjet2 = recocalojet->et();
	etajet2 = recocalojet->eta();
      }
      countjets++;
    }
    float deltaetajet = etajet1 - etajet2;

    if(etjet1>minEt_ && etjet2>minEt_ && (etajet1*etajet2)<0 && fabs(deltaetajet)>minDeltaEta_){

      for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	   recocalojet<=(recocalojets->begin()+1); recocalojet++) {
	ref=edm::RefToBase<reco::Candidate>(reco::CaloJetRef(recocalojets,
							     distance(recocalojets->begin(),recocalojet)));
	filterproduct->putParticle(ref);
	n++;
      }
    }
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>=2);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}

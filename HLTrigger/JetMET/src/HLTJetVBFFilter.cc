/** \class HLTJetVBFFilter
 *
 * $Id: HLTJetVBFFilter.cc,v 1.5 2008/06/07 19:46:08 apana Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTJetVBFFilter::HLTJetVBFFilter(const edm::ParameterSet& iConfig)
{
   inputTag_    = iConfig.getParameter< edm::InputTag > ("inputTag");
   saveTag_     = iConfig.getUntrackedParameter<bool>("saveTag",false);
   minEtLow_    = iConfig.getParameter<double> ("minEtLow");
   minEtHigh_   = iConfig.getParameter<double> ("minEtHigh");
   minDeltaEta_ = iConfig.getParameter<double> ("minDeltaEta"); 

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTJetVBFFilter::~HLTJetVBFFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTJetVBFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace trigger;
  // The filter object
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTag_) filterobject->addCollectionTag(inputTag_);

  edm::Handle<reco::CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputTag_,recocalojets);

  // look at all candidates, check cuts and add to filter object
  int n(0);
  
  
  // events with two or more jets
  if(recocalojets->size() > 1){
    
    double etjet1 = 0.;
    double etjet2 = 0.;
    double etajet1 = 0.;
    double etajet2 = 0.;
    
    // loop on all jets
    for (reco::CaloJetCollection::const_iterator recocalojet1 = recocalojets->begin(); 
         recocalojet1 != recocalojets->end(); ++recocalojet1) {
      
      if( recocalojet1->et() < minEtHigh_ ) break;
      
      for (reco::CaloJetCollection::const_iterator recocalojet2 = recocalojet1+1; 
           recocalojet2 != recocalojets->end(); ++recocalojet2) {
        
        if( recocalojet2->et() < minEtLow_ ) break;
        
        etjet1 = recocalojet1->et();
	etajet1 = recocalojet1->eta();
        
        etjet2 = recocalojet2->et();
        etajet2 = recocalojet2->eta();
        
        float deltaetajet = etajet1 - etajet2;
        
        // VBF cuts
        if ( (etjet1 > minEtHigh_) &&
             (etjet2 > minEtLow_) &&
             (etajet1*etajet2 < 0 ) &&
             (fabs(deltaetajet) > minDeltaEta_) ){
          
   	  ++n;
          reco::CaloJetRef ref1(reco::CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet1)));
          filterobject->addObject(TriggerJet,ref1);
          reco::CaloJetRef ref2(reco::CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet2)));
          filterobject->addObject(TriggerJet,ref2);
        
        } // VBF cuts
      
      }
    } // loop on all jets
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>=1);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}

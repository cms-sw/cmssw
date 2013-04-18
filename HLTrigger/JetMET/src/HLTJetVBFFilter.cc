/** \class HLTJetVBFFilter
 *
 * $Id: HLTJetVBFFilter.cc,v 1.6 2011/02/09 06:38:58 gruen Exp $
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
   etaOpposite_ = iConfig.getParameter<bool>   ("etaOpposite"); 
   minDeltaEta_ = iConfig.getParameter<double> ("minDeltaEta"); 
   minInvMass_  = iConfig.getParameter<double> ("minInvMass"); 

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
    
    double ejet1 = 0.;
    double pxjet1 = 0.;
    double pyjet1 = 0.;
    double pzjet1 = 0.;
    double etjet1 = 0.;
    double etajet1 = 0.;
    
    double ejet2 = 0.;
    double pxjet2 = 0.;
    double pyjet2 = 0.;
    double pzjet2 = 0.;
    double etjet2 = 0.;
    double etajet2 = 0.;
    
    // loop on all jets
    for (reco::CaloJetCollection::const_iterator recocalojet1 = recocalojets->begin(); 
         recocalojet1 != recocalojets->end(); ++recocalojet1) {
      
      if( recocalojet1->et() < minEtHigh_ ) break;
      
      for (reco::CaloJetCollection::const_iterator recocalojet2 = recocalojet1+1; 
           recocalojet2 != recocalojets->end(); ++recocalojet2) {
        
        if( recocalojet2->et() < minEtLow_ ) break;
        
        ejet1 = recocalojet1->energy();
        pxjet1 = recocalojet1->px();
        pyjet1 = recocalojet1->py();
        pzjet1 = recocalojet1->pz();
        etjet1 = recocalojet1->et();
	etajet1 = recocalojet1->eta();

        ejet2 = recocalojet2->energy();
        pxjet2 = recocalojet2->px();
        pyjet2 = recocalojet2->py();
        pzjet2 = recocalojet2->pz();
        etjet2 = recocalojet2->et();
        etajet2 = recocalojet2->eta();
        
        float deltaetajet = etajet1 - etajet2;
        
        float invmassjet = sqrt( (ejet1  + ejet2)  * (ejet1  + ejet2) - 
      	                         (pxjet1 + pxjet2) * (pxjet1 + pxjet2) - 
                                 (pyjet1 + pyjet2) * (pyjet1 + pyjet2) - 
                                 (pzjet1 + pzjet2) * (pzjet1 + pzjet2) );
        
        // VBF cuts
        if ( (etjet1 > minEtHigh_) &&
             (etjet2 > minEtLow_) &&
             ( (etaOpposite_ == true && etajet1*etajet2 < 0) || (etaOpposite_ == false) ) &&
             (fabs(deltaetajet) > minDeltaEta_) &&
	     (fabs(invmassjet) > minInvMass_) ){
          
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

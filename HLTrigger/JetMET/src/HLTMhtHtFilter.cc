/** \class HLTMhtHtFilter
 *
 *
 *  \author Gheorghe Lungu
 *
 */

#include "HLTrigger/JetMET/interface/HLTMhtHtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaPhi.h"

//
// constructors and destructor
//
HLTMhtHtFilter::HLTMhtHtFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   saveTag_     = iConfig.getUntrackedParameter<bool>("saveTag",false);
   minMht_= iConfig.getParameter<double> ("minMht"); 
   minPtJet_= iConfig.getParameter<double> ("minPtJet"); 

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMhtHtFilter::~HLTMhtHtFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTMhtHtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTag_) filterobject->addCollectionTag(inputJetTag_);

  CaloJetRef ref;
  // Get the Candidates

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(recocalojets->size() > 0){
    // events with at least one jet

    double mhtx=0., mhty=0.;


    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet != recocalojets->end(); recocalojet++) {

      if (recocalojet->pt() > minPtJet_) {
	mhtx -= recocalojet->pt()*cos(recocalojet->phi());
	mhty -= recocalojet->pt()*sin(recocalojet->phi());
      }
    }
    
    
    if( sqrt(mhtx*mhtx + mhty*mhty) > minMht_){

      for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); recocalojet!=recocalojets->end(); recocalojet++) {
	ref = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
	filterobject->addObject(TriggerJet,ref);
	n++;
      }
    }
    
  } // events with at least one jet
  
  
  
  // filter decision
  bool accept(n>0);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}

/** \class HLTNVFilter
 *
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/JetMET/interface/HLTNVFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTNVFilter::HLTNVFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   inputMETTag_ = iConfig.getParameter< edm::InputTag > ("inputMETTag");
   saveTags_    = iConfig.getParameter<bool>("saveTags");
   minNV_   = iConfig.getParameter<double> ("minNV");
   minEtjet1_= iConfig.getParameter<double> ("minEtJet1"); 
   minEtjet2_ = iConfig.getParameter<double> ("minEtJet2");
   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTNVFilter::~HLTNVFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTNVFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) {
    filterobject->addCollectionTag(inputJetTag_);
    filterobject->addCollectionTag(inputMETTag_);
  }

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);
  Handle<trigger::TriggerFilterObjectWithRefs> metcal;
  iEvent.getByLabel(inputMETTag_,metcal);


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(recocalojets->size() > 1){
    // events with two or more jets

    double etjet1=0.;
    double etjet2=0.;
    double etmiss=0.;
    int countjets =0;
   
    VRcalomet vrefMET; 
    metcal->getObjects(TriggerMET,vrefMET);
    CaloMETRef metRef=vrefMET.at(0);
    etmiss=vrefMET.at(0)->et();

    CaloJetRef ref1,ref2;
    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet<=(recocalojets->begin()+1); recocalojet++) {
      
      if(countjets==0) {
	etjet1 = recocalojet->et();
                ref1  = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
      }
      if(countjets==1) {
	etjet2 = recocalojet->et();
                ref2  = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
      }
      countjets++;
    }
    
    double NV = (etmiss*etmiss-(etjet1-etjet2)*(etjet1-etjet2))/(etjet2*etjet2);
    if(etjet1>minEtjet1_  && etjet2>minEtjet2_ && NV>minNV_){
      filterobject->addObject(TriggerMET,metRef);
      filterobject->addObject(TriggerJet,ref1);
      filterobject->addObject(TriggerJet,ref2);
      n++;
    }
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>=1);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}

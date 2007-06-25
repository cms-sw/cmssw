#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"
void PFRecoTauProducer::produce(Event& iEvent, const EventSetup& iSetup){

  auto_ptr<TauCollection> resultTau(new TauCollection);

  Handle<PFIsolatedTauTagInfoCollection> thePFTagInfo;
  iEvent.getByLabel(PFTagInfo_,thePFTagInfo);
  const PFIsolatedTauTagInfoCollection& myPFTagInfo=*(thePFTagInfo.product()); 

  for(PFIsolatedTauTagInfoCollection::const_iterator i_GJ=myPFTagInfo.begin();i_GJ!=myPFTagInfo.end();i_GJ++) { 
    Tau myTau=PFRecoTauAlgo_->tag(*i_GJ);
    resultTau->push_back(myTau);
  }
  
  iEvent.put(resultTau);
}

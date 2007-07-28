#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"
void PFRecoTauProducer::produce(Event& iEvent, const EventSetup& iSetup){

  auto_ptr<TauCollection> resultTau(new TauCollection);

  Handle<PFIsolatedTauTagInfoCollection> thePFTagInfo;
  iEvent.getByLabel(PFTagInfo_,thePFTagInfo);
  const PFIsolatedTauTagInfoCollection& myPFTagInfo=*(thePFTagInfo.product()); 
  for(PFIsolatedTauTagInfoCollection::const_iterator i_info=myPFTagInfo.begin();i_info!=myPFTagInfo.end();i_info++) { 
    if((*i_info).pfjetRef()->pt()>JetMinPt_){ 
      Tau myTau=PFRecoTauAlgo_->tag(*i_info);
      resultTau->push_back(myTau);
    }
  }
  iEvent.put(resultTau);
}

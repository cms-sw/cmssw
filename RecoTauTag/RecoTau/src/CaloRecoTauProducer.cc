#include "RecoTauTag/RecoTau/interface/CaloRecoTauProducer.h"
void CaloRecoTauProducer::produce(Event& iEvent, const EventSetup& iSetup){

  auto_ptr<TauCollection> resultTau(new TauCollection);

  Handle<CombinedTauTagInfoCollection> theCaloTagInfo;
  iEvent.getByLabel(CaloTagInfo_,theCaloTagInfo);
  const CombinedTauTagInfoCollection& myCaloTagInfo=*(theCaloTagInfo.product()); 
  for(CombinedTauTagInfoCollection::const_iterator i_GJ=myCaloTagInfo.begin();i_GJ!=myCaloTagInfo.end();i_GJ++) { 
    if((i_GJ->jet())->pt() > JetMinPt_){ 
      Tau myTau=CaloRecoTauAlgo_->tag(*i_GJ);
      resultTau->push_back(myTau);
    }
  }
  
  iEvent.put(resultTau);
}

#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"
void PFRecoTauProducer::produce(Event& iEvent,const EventSetup& iSetup){
  auto_ptr<TauCollection> resultTau(new TauCollection);

  ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
  PFRecoTauAlgo_->setTransientTrackBuilder(myTransientTrackBuilder.product());
  
  Handle<VertexCollection> thePVs;
  iEvent.getByLabel(PVProducer_,thePVs);
  Vertex thePV;
  if((*thePVs).size()) thePV=*((*thePVs).begin());
  else{
    Vertex::Error e;
    e(0,0)=0.0015*0.0015;
    e(1,1)=0.0015*0.0015;
    e(2,2)=0.0050*0.0050;
    Vertex::Point p(0,0,0);
    thePV=Vertex(p,e,1,1,1);
  }

  Handle<PFIsolatedTauTagInfoCollection> thePFTagInfo;
  iEvent.getByLabel(PFTagInfo_,thePFTagInfo);
  const PFIsolatedTauTagInfoCollection& myPFTagInfo=*(thePFTagInfo.product()); 
  for(PFIsolatedTauTagInfoCollection::const_iterator i_info=myPFTagInfo.begin();i_info!=myPFTagInfo.end();i_info++) { 
    if((*i_info).pfjetRef()->pt()>JetMinPt_){ 
      Tau myTau=PFRecoTauAlgo_->tag(*i_info,thePV);
      resultTau->push_back(myTau);
    }
  }
  iEvent.put(resultTau);
}

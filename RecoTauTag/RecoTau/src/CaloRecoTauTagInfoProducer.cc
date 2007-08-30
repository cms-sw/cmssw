#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoProducer.h"
#include <Math/GenVector/VectorUtil.h>

void CaloRecoTauTagInfoProducer::produce(Event& iEvent, const EventSetup& iSetup){
  Handle<JetTracksAssociationCollection> theCaloJetTracksAssociatorCollection;
  iEvent.getByLabel(CaloJetTracksAssociatormodule_,theCaloJetTracksAssociatorCollection);
  
  // query a sim/rec PV
  Vertex::Error PVError;
  PVError(0,0)=smearedPVsigmaX_*smearedPVsigmaX_;
  PVError(1,1)=smearedPVsigmaY_*smearedPVsigmaY_;
  PVError(2,2)=smearedPVsigmaZ_*smearedPVsigmaZ_;
  Vertex::Point PVPoint(0,0,0);
  Handle<SimVertexContainer> G4VtxContainer;
  iEvent.getByType(G4VtxContainer);
  if (G4VtxContainer.isValid() && G4VtxContainer->size()) {
    Vertex::Point SimPVPoint(RandGauss::shoot(G4VtxContainer->begin()->position().x(),smearedPVsigmaX_),  
			     RandGauss::shoot(G4VtxContainer->begin()->position().y(),smearedPVsigmaY_),  
			     RandGauss::shoot(G4VtxContainer->begin()->position().z(),smearedPVsigmaZ_));
    PVPoint=SimPVPoint;
  }
  Vertex myPV(PVPoint,PVError,1,1,1);
  
  Handle<VertexCollection> vertices;
  iEvent.getByLabel(PVmodule_,vertices);
  const VertexCollection vertCollection=*(vertices.product());
  if(vertCollection.size()) myPV=*(vertCollection.begin());
  
  TauTagInfoCollection* extCollection=new TauTagInfoCollection();

  unsigned int i_Assoc=0;
  for(JetTracksAssociationCollection::const_iterator iAssoc=theCaloJetTracksAssociatorCollection->begin();iAssoc!=theCaloJetTracksAssociatorCollection->end();iAssoc++){
    TauTagInfo myTauTagInfo=CaloRecoTauTagInfoAlgo_->tag((*iAssoc).first.castTo<CaloJetRef>(),(*iAssoc).second,myPV);
    extCollection->push_back(myTauTagInfo);
    ++i_Assoc;
  }
  
  auto_ptr<TauTagInfoCollection> resultExt(extCollection);  
  iEvent.put(resultExt);  
}

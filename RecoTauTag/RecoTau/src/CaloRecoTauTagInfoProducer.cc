#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoProducer.h"

CaloRecoTauTagInfoProducer::CaloRecoTauTagInfoProducer(const ParameterSet& iConfig){
  CaloJetTracksAssociatorProducer_ = iConfig.getParameter<string>("CaloJetTracksAssociatorProducer");
  PVProducer_                    = iConfig.getParameter<string>("PVProducer");
  smearedPVsigmaX_               = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_               = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_               = iConfig.getParameter<double>("smearedPVsigmaZ");	
  CaloRecoTauTagInfoAlgo_=new CaloRecoTauTagInfoAlgorithm(iConfig);
  produces<CaloTauTagInfoCollection>();      
}
CaloRecoTauTagInfoProducer::~CaloRecoTauTagInfoProducer(){
  delete CaloRecoTauTagInfoAlgo_;
}

void CaloRecoTauTagInfoProducer::produce(Event& iEvent,const EventSetup& iSetup){
  Handle<JetTracksAssociationCollection> theCaloJetTracksAssociatorCollection;
  iEvent.getByLabel(CaloJetTracksAssociatorProducer_,theCaloJetTracksAssociatorCollection);
  
  // query a rec/sim PV
  Handle<VertexCollection> thePVs;
  iEvent.getByLabel(PVProducer_,thePVs);
  const VertexCollection vertCollection=*(thePVs.product());
  Vertex thePV;
  if(vertCollection.size()) thePV=*(vertCollection.begin());
  else{
    Vertex::Error SimPVError;
    SimPVError(0,0)=smearedPVsigmaX_*smearedPVsigmaX_;
    SimPVError(1,1)=smearedPVsigmaY_*smearedPVsigmaY_;
    SimPVError(2,2)=smearedPVsigmaZ_*smearedPVsigmaZ_;
    Vertex::Point SimPVPoint(RandGauss::shoot(0.,smearedPVsigmaX_),  
			     RandGauss::shoot(0.,smearedPVsigmaY_),  
			     RandGauss::shoot(0.,smearedPVsigmaZ_));
    thePV=Vertex(SimPVPoint,SimPVError,1,1,1);    
  }
  
  CaloTauTagInfoCollection* extCollection=new CaloTauTagInfoCollection();

  for(JetTracksAssociationCollection::const_iterator iAssoc=theCaloJetTracksAssociatorCollection->begin();iAssoc!=theCaloJetTracksAssociatorCollection->end();iAssoc++){
    CaloTauTagInfo myCaloTauTagInfo=CaloRecoTauTagInfoAlgo_->buildCaloTauTagInfo(iEvent,iSetup,(*iAssoc).first.castTo<CaloJetRef>(),(*iAssoc).second,thePV);
    extCollection->push_back(myCaloTauTagInfo);
  }
  
  auto_ptr<CaloTauTagInfoCollection> resultExt(extCollection);  
  iEvent.put(resultExt);  
}

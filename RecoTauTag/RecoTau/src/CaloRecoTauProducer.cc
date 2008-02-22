#include "RecoTauTag/RecoTau/interface/CaloRecoTauProducer.h"

CaloRecoTauProducer::CaloRecoTauProducer(const ParameterSet& iConfig){
  CaloRecoTauTagInfoProducer_  = iConfig.getParameter<InputTag>("CaloRecoTauTagInfoProducer");
  PVProducer_                  = iConfig.getParameter<string>("PVProducer");
  smearedPVsigmaX_             = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_             = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_             = iConfig.getParameter<double>("smearedPVsigmaZ");	
  JetMinPt_                    = iConfig.getParameter<double>("JetPtMin");
  CaloRecoTauAlgo_=new CaloRecoTauAlgorithm(iConfig);
  produces<CaloTauCollection>();      
}
CaloRecoTauProducer::~CaloRecoTauProducer(){
  delete CaloRecoTauAlgo_;
}
  
void CaloRecoTauProducer::produce(Event& iEvent,const EventSetup& iSetup){

  auto_ptr<CaloTauCollection> resultCaloTau(new CaloTauCollection);

  ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
  CaloRecoTauAlgo_->setTransientTrackBuilder(myTransientTrackBuilder.product());
  
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
  
  Handle<CaloTauTagInfoCollection> theCaloTauTagInfoCollection;
  iEvent.getByLabel(CaloRecoTauTagInfoProducer_,theCaloTauTagInfoCollection);
  int iinfo=0;
  for(CaloTauTagInfoCollection::const_iterator i_info=theCaloTauTagInfoCollection->begin();i_info!=theCaloTauTagInfoCollection->end();i_info++) { 
    if(i_info->calojetRef()->pt()>JetMinPt_){ 
      CaloTau myCaloTau=CaloRecoTauAlgo_->buildCaloTau(iEvent,Ref<CaloTauTagInfoCollection>(theCaloTauTagInfoCollection,iinfo),thePV);
      resultCaloTau->push_back(myCaloTau);
    }
    ++iinfo;
  }
  
  iEvent.put(resultCaloTau);
}

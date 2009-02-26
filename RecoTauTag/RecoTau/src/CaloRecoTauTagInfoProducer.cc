#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoProducer.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"


CaloRecoTauTagInfoProducer::CaloRecoTauTagInfoProducer(const ParameterSet& iConfig){
  CaloJetTracksAssociatorProducer_ = iConfig.getParameter<InputTag>("CaloJetTracksAssociatorProducer");
  PVProducer_                    = iConfig.getParameter<InputTag>("PVProducer");
  smearedPVsigmaX_               = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_               = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_               = iConfig.getParameter<double>("smearedPVsigmaZ");	
  CaloRecoTauTagInfoAlgo_=new CaloRecoTauTagInfoAlgorithm(iConfig);

  produces<CaloTauTagInfoCollection>();  
  //produces<DetIdCollection>();
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
  thePV=*(vertCollection.begin());
  
  //  auto_ptr<DetIdCollection> selectedDetIds(new DetIdCollection);
  CaloTauTagInfoCollection* extCollection=new CaloTauTagInfoCollection();

  for(JetTracksAssociationCollection::const_iterator iAssoc=theCaloJetTracksAssociatorCollection->begin();iAssoc!=theCaloJetTracksAssociatorCollection->end();iAssoc++){
    CaloTauTagInfo myCaloTauTagInfo=CaloRecoTauTagInfoAlgo_->buildCaloTauTagInfo(iEvent,iSetup,(*iAssoc).first.castTo<CaloJetRef>(),(*iAssoc).second,thePV);
    extCollection->push_back(myCaloTauTagInfo);
    //    vector<DetId> myDets = CaloRecoTauTagInfoAlgo_->getVectorDetId((*iAssoc).first.castTo<CaloJetRef>());

      //Saving the selectedDetIds
    //    for(unsigned int i=0; i<myDets.size();i++)
    //      selectedDetIds->push_back(myDets[i]);
  }
  
  auto_ptr<CaloTauTagInfoCollection> resultExt(extCollection);  
  iEvent.put(resultExt);  
  //  iEvent.put(selectedDetIds);
}

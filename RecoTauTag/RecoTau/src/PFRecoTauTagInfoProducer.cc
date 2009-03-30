#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoProducer.h"

PFRecoTauTagInfoProducer::PFRecoTauTagInfoProducer(const ParameterSet& iConfig){

  PFCandidateProducer_                = iConfig.getParameter<edm::InputTag>("PFCandidateProducer");
  PFJetTracksAssociatorProducer_      = iConfig.getParameter<edm::InputTag>("PFJetTracksAssociatorProducer");
  PVProducer_                         = iConfig.getParameter<edm::InputTag>("PVProducer");
  smearedPVsigmaX_                    = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_                    = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_                    = iConfig.getParameter<double>("smearedPVsigmaZ");	
  PFRecoTauTagInfoAlgo_=new PFRecoTauTagInfoAlgorithm(iConfig);
  produces<PFTauTagInfoCollection>();      
}
PFRecoTauTagInfoProducer::~PFRecoTauTagInfoProducer(){
  delete PFRecoTauTagInfoAlgo_;
}

void PFRecoTauTagInfoProducer::produce(Event& iEvent, const EventSetup& iSetup){
  Handle<JetTracksAssociationCollection> thePFJetTracksAssociatorCollection;
  iEvent.getByLabel(PFJetTracksAssociatorProducer_,thePFJetTracksAssociatorCollection);
  // *** access the PFCandidateCollection in the event in order to retrieve the PFCandidateRefVector which constitutes each PFJet
  Handle<PFCandidateCollection> thePFCandidateCollection;
  iEvent.getByLabel(PFCandidateProducer_,thePFCandidateCollection);
  PFCandidateRefVector thePFCandsInTheEvent;
  for(unsigned int i_PFCand=0;i_PFCand!=thePFCandidateCollection->size();i_PFCand++) { 
        thePFCandsInTheEvent.push_back(PFCandidateRef(thePFCandidateCollection,i_PFCand));
  }
  // ***
  // query a rec/sim PV
  Handle<VertexCollection> thePVs;
  iEvent.getByLabel(PVProducer_,thePVs);
  const VertexCollection vertCollection=*(thePVs.product());
  math::XYZPoint V(0,0,-1000.);

  Vertex thePV;
  if(vertCollection.size() > 0) thePV =*(vertCollection.begin());
else{
    Vertex::Error SimPVError;
    SimPVError(0,0)=15.*15.;
    SimPVError(1,1)=15.*15.;
    SimPVError(2,2)=15.*15.;
    Vertex::Point SimPVPoint(0.,0.,-1000.);
    thePV=Vertex(SimPVPoint,SimPVError,1,1,1);    
  }
  
  auto_ptr<PFTauTagInfoCollection> resultExt(new PFTauTagInfoCollection);  
  for(JetTracksAssociationCollection::const_iterator iAssoc=thePFJetTracksAssociatorCollection->begin();iAssoc!=thePFJetTracksAssociatorCollection->end();iAssoc++){
    PFTauTagInfo myPFTauTagInfo=PFRecoTauTagInfoAlgo_->buildPFTauTagInfo((*iAssoc).first.castTo<PFJetRef>(),thePFCandsInTheEvent,(*iAssoc).second,thePV);
    resultExt->push_back(myPFTauTagInfo);
  }
  

  //  OrphanHandle<PFTauTagInfoCollection> myPFTauTagInfoCollection=iEvent.put(resultExt);
  iEvent.put(resultExt);
}

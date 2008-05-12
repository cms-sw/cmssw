#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"

PFRecoTauProducer::PFRecoTauProducer(const ParameterSet& iConfig){
  PFTauTagInfoProducer_   = iConfig.getParameter<InputTag>("PFTauTagInfoProducer");
  ElectronPreIDProducer_  = iConfig.getParameter<InputTag>("ElectronPreIDProducer");
  PVProducer_             = iConfig.getParameter<string>("PVProducer");
  smearedPVsigmaX_        = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_        = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_        = iConfig.getParameter<double>("smearedPVsigmaZ");	
  JetMinPt_               = iConfig.getParameter<double>("JetPtMin");
  PFRecoTauAlgo_=new PFRecoTauAlgorithm(iConfig);
  produces<PFTauCollection>();      
}
PFRecoTauProducer::~PFRecoTauProducer(){
  delete PFRecoTauAlgo_;
}

void PFRecoTauProducer::produce(Event& iEvent,const EventSetup& iSetup){
  auto_ptr<PFTauCollection> resultPFTau(new PFTauCollection);
  
  ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
  PFRecoTauAlgo_->setTransientTrackBuilder(myTransientTrackBuilder.product());

  ESHandle<MagneticField> myMF;
  iSetup.get<IdealMagneticFieldRecord>().get(myMF);
  PFRecoTauAlgo_->setMagneticField(myMF.product());

  // Electron PreID tracks: Temporary until integrated to PFCandidate
  edm::Handle<PFRecTrackCollection> myPFelecTk; 
  iEvent.getByLabel(ElectronPreIDProducer_,myPFelecTk); 
  const PFRecTrackCollection theElecTkCollection=*(myPFelecTk.product()); 
  
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
  
  Handle<PFTauTagInfoCollection> thePFTauTagInfoCollection;
  iEvent.getByLabel(PFTauTagInfoProducer_,thePFTauTagInfoCollection);
  int iinfo=0;
  for(PFTauTagInfoCollection::const_iterator i_info=thePFTauTagInfoCollection->begin();i_info!=thePFTauTagInfoCollection->end();i_info++) { 
    if((*i_info).pfjetRef()->pt()>JetMinPt_){
      PFTau myPFTau=PFRecoTauAlgo_->buildPFTau(Ref<PFTauTagInfoCollection>(thePFTauTagInfoCollection,iinfo),thePV,theElecTkCollection);
      resultPFTau->push_back(myPFTau);
    }
    ++iinfo;
  }
  iEvent.put(resultPFTau);
}

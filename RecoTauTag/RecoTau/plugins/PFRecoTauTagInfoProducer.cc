/* class PFRecoTauTagInfoProducer
 * returns a PFTauTagInfo collection starting from a JetTrackAssociations <a PFJet,a list of Tracks> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGauss.h"

#include "Math/GenVector/VectorUtil.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauTagInfoProducer : public EDProducer {
 public:
  explicit PFRecoTauTagInfoProducer(const edm::ParameterSet& iConfig);
  ~PFRecoTauTagInfoProducer();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  PFRecoTauTagInfoAlgorithm* PFRecoTauTagInfoAlgo_;
  edm::InputTag PFCandidateProducer_;
  edm::InputTag PFJetTracksAssociatorProducer_;
  edm::InputTag PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};

PFRecoTauTagInfoProducer::PFRecoTauTagInfoProducer(const edm::ParameterSet& iConfig){

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

void PFRecoTauTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::Handle<JetTracksAssociationCollection> thePFJetTracksAssociatorCollection;
  iEvent.getByLabel(PFJetTracksAssociatorProducer_,thePFJetTracksAssociatorCollection);
  // *** access the PFCandidateCollection in the event in order to retrieve the PFCandidateRefVector which constitutes each PFJet
  edm::Handle<PFCandidateCollection> thePFCandidateCollection;
  iEvent.getByLabel(PFCandidateProducer_,thePFCandidateCollection);
  vector<PFCandidatePtr> thePFCandsInTheEvent;
  for(unsigned int i_PFCand=0;i_PFCand!=thePFCandidateCollection->size();i_PFCand++) { 
        thePFCandsInTheEvent.push_back(PFCandidatePtr(thePFCandidateCollection,i_PFCand));
  }
  // ***
  // query a rec/sim PV
  edm::Handle<VertexCollection> thePVs;
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
DEFINE_FWK_MODULE(PFRecoTauTagInfoProducer);

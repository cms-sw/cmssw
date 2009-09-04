/* class CaloRecoTauTagInfoProducer 
 * returns a CaloTauTagInfo collection starting from a JetTrackAssociations <a CaloJet,a list of Track's> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "CLHEP/Random/RandGauss.h"

#include "Math/GenVector/VectorUtil.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class CaloRecoTauTagInfoProducer : public EDProducer {
 public:
  explicit CaloRecoTauTagInfoProducer(const ParameterSet&);
  ~CaloRecoTauTagInfoProducer();
  virtual void produce(Event&,const EventSetup&);
 private:
  CaloRecoTauTagInfoAlgorithm* CaloRecoTauTagInfoAlgo_;
  InputTag CaloJetTracksAssociatorProducer_;
  InputTag PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};


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
DEFINE_FWK_MODULE(CaloRecoTauTagInfoProducer );

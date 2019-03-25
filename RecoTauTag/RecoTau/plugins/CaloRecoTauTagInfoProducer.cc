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
  explicit CaloRecoTauTagInfoProducer(const edm::ParameterSet&);
  ~CaloRecoTauTagInfoProducer() override;
  void produce(edm::Event&,const edm::EventSetup&) override;
 private:
  CaloRecoTauTagInfoAlgorithm* CaloRecoTauTagInfoAlgo_;
  edm::InputTag CaloJetTracksAssociatorProducer_;
  edm::InputTag PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};


CaloRecoTauTagInfoProducer::CaloRecoTauTagInfoProducer(const edm::ParameterSet& iConfig){
  CaloJetTracksAssociatorProducer_ = iConfig.getParameter<edm::InputTag>("CaloJetTracksAssociatorProducer");
  PVProducer_                    = iConfig.getParameter<edm::InputTag>("PVProducer");
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

void CaloRecoTauTagInfoProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  edm::Handle<JetTracksAssociationCollection> theCaloJetTracksAssociatorCollection;
  iEvent.getByLabel(CaloJetTracksAssociatorProducer_,theCaloJetTracksAssociatorCollection);
  
  // query a rec/sim PV
  edm::Handle<VertexCollection> thePVs;
  iEvent.getByLabel(PVProducer_,thePVs);
  const VertexCollection vertCollection=*(thePVs.product());
  Vertex thePV;
  thePV=*(vertCollection.begin());
  
  //  auto selectedDetIds = std::make_unique<DetIdCollection>();
  CaloTauTagInfoCollection* extCollection=new CaloTauTagInfoCollection();

  for(JetTracksAssociationCollection::const_iterator iAssoc=theCaloJetTracksAssociatorCollection->begin();iAssoc!=theCaloJetTracksAssociatorCollection->end();iAssoc++){
//    CaloTauTagInfo myCaloTauTagInfo=CaloRecoTauTagInfoAlgo_->buildCaloTauTagInfo(iEvent,iSetup,(*iAssoc).first.castTo<CaloJetRef>(),(*iAssoc).second,thePV);
    CaloTauTagInfo myCaloTauTagInfo=CaloRecoTauTagInfoAlgo_->buildCaloTauTagInfo(iEvent,iSetup,(*iAssoc).first,(*iAssoc).second,thePV);
    extCollection->push_back(myCaloTauTagInfo);
    //    std::vector<DetId> myDets = CaloRecoTauTagInfoAlgo_->getVectorDetId((*iAssoc).first.castTo<CaloJetRef>());

      //Saving the selectedDetIds
    //    for(unsigned int i=0; i<myDets.size();i++)
    //      selectedDetIds->push_back(myDets[i]);
  }
  
  std::unique_ptr<CaloTauTagInfoCollection> resultExt(extCollection);  
  iEvent.put(std::move(resultExt));  

  //  iEvent.put(std::move(selectedDetIds));
}
DEFINE_FWK_MODULE(CaloRecoTauTagInfoProducer );

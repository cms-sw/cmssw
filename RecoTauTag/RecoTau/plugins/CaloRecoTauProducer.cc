/* class CaloRecoTauProducer
 * EDProducer of the CaloTauCollection, starting from the CaloTauTagInfoCollection, 
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "CLHEP/Random/RandGauss.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class CaloRecoTauProducer : public EDProducer {
 public:
  explicit CaloRecoTauProducer(const edm::ParameterSet& iConfig);
  ~CaloRecoTauProducer();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  edm::InputTag CaloRecoTauTagInfoProducer_;
  edm::InputTag PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;
  double JetMinPt_;
  CaloRecoTauAlgorithm* CaloRecoTauAlgo_;
};

CaloRecoTauProducer::CaloRecoTauProducer(const edm::ParameterSet& iConfig){
  CaloRecoTauTagInfoProducer_  = iConfig.getParameter<edm::InputTag>("CaloRecoTauTagInfoProducer");
  PVProducer_                  = iConfig.getParameter<edm::InputTag>("PVProducer");
  smearedPVsigmaX_             = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_             = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_             = iConfig.getParameter<double>("smearedPVsigmaZ");	
  JetMinPt_                    = iConfig.getParameter<double>("JetPtMin");
  CaloRecoTauAlgo_=new CaloRecoTauAlgorithm(iConfig);
  produces<CaloTauCollection>();
  produces<DetIdCollection>();
}
CaloRecoTauProducer::~CaloRecoTauProducer(){
  delete CaloRecoTauAlgo_;
}
  
void CaloRecoTauProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){

  auto_ptr<CaloTauCollection> resultCaloTau(new CaloTauCollection);
  auto_ptr<DetIdCollection> selectedDetIds(new DetIdCollection);
 
  edm::ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
  CaloRecoTauAlgo_->setTransientTrackBuilder(myTransientTrackBuilder.product());
  
  edm::ESHandle<MagneticField> myMF;
  iSetup.get<IdealMagneticFieldRecord>().get(myMF);
  CaloRecoTauAlgo_->setMagneticField(myMF.product());
    
  // query a rec/sim PV
  edm::Handle<VertexCollection> thePVs;
  iEvent.getByLabel(PVProducer_,thePVs);
  const VertexCollection vertCollection=*(thePVs.product());
  Vertex thePV;
  if(vertCollection.size()) thePV=*(vertCollection.begin());
  else{
    Vertex::Error SimPVError;
    SimPVError(0,0)=smearedPVsigmaX_*smearedPVsigmaX_;
    SimPVError(1,1)=smearedPVsigmaY_*smearedPVsigmaY_;
    SimPVError(2,2)=smearedPVsigmaZ_*smearedPVsigmaZ_;
    Vertex::Point SimPVPoint(CLHEP::RandGauss::shoot(0.,smearedPVsigmaX_),  
			     CLHEP::RandGauss::shoot(0.,smearedPVsigmaY_),  
			     CLHEP::RandGauss::shoot(0.,smearedPVsigmaZ_));
    thePV=Vertex(SimPVPoint,SimPVError,1,1,1);    
  }
  
  edm::Handle<CaloTauTagInfoCollection> theCaloTauTagInfoCollection;
  iEvent.getByLabel(CaloRecoTauTagInfoProducer_,theCaloTauTagInfoCollection);
  int iinfo=0;
  for(CaloTauTagInfoCollection::const_iterator i_info=theCaloTauTagInfoCollection->begin();i_info!=theCaloTauTagInfoCollection->end();i_info++) { 
    if(i_info->jetRef()->pt()>JetMinPt_){ 
      CaloTau myCaloTau=CaloRecoTauAlgo_->buildCaloTau(iEvent,iSetup,Ref<CaloTauTagInfoCollection>(theCaloTauTagInfoCollection,iinfo),thePV);
      resultCaloTau->push_back(myCaloTau);
    }
    ++iinfo;
  }
  for(unsigned int i =0;i<CaloRecoTauAlgo_->mySelectedDetId_.size();i++)
    selectedDetIds->push_back(CaloRecoTauAlgo_->mySelectedDetId_[i]);


   iEvent.put(resultCaloTau);
  iEvent.put(selectedDetIds);
}
DEFINE_FWK_MODULE(CaloRecoTauProducer);

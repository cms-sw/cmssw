#include "Calibration/HcalCalibAlgos/src/HitReCalibrator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "FWCore/Utilities/interface/Exception.h"


using namespace edm;
using namespace std;
using namespace reco;

namespace cms
{

HitReCalibrator::HitReCalibrator(const edm::ParameterSet& iConfig)
{
   hbheInput_ = iConfig.getParameter<edm::InputTag>("hbheInput");
   hoInput_ = iConfig.getParameter<edm::InputTag>("hoInput");
   hfInput_ = iConfig.getParameter<edm::InputTag>("hfInput"); 
   allowMissingInputs_ = true;
//register your products

   produces<HBHERecHitCollection>("DiJetsHBHEReRecHitCollection");
   produces<HORecHitCollection>("DiJetsHOReRecHitCollection");
   produces<HFRecHitCollection>("DiJetsHFReRecHitCollection");

}
void HitReCalibrator::beginJob()
{
}

HitReCalibrator::~HitReCalibrator()
{

}


// ------------ method called to produce the data  ------------
void
HitReCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<HBHERecHitCollection> miniDiJetsHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniDiJetsHORecHitCollection(new HORecHitCollection);
  std::auto_ptr<HFRecHitCollection> miniDiJetsHFRecHitCollection(new HFRecHitCollection);

 
  edm::ESHandle <HcalRespCorrs> recalibCorrs;
  iSetup.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
  const HcalRespCorrs* jetRecalib = recalibCorrs.product();
 
   
  try {
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel(hbheInput_,hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());
  for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
        {
           DetId id = hbheItr->detid();
           float recal; 
           if (jetRecalib->exists(id))
             recal = jetRecalib->getValues(id)->getValue();
	   else recal = 1.; 
           float energy = hbheItr->energy(); 
           float time = hbheItr->time();
           HBHERecHit* hbhehit = new HBHERecHit(id,recal*energy,time);  
           miniDiJetsHBHERecHitCollection->push_back(*hbhehit);
        }
  } catch (cms::Exception& e) { // can't find it!
  if (!allowMissingInputs_) {cout<<"No HBHE collection "<<endl; throw e;}
  }

  try{  
  edm::Handle<HORecHitCollection> ho;
  iEvent.getByLabel(hoInput_,ho);
  const HORecHitCollection Hitho = *(ho.product());
  for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
        {
           DetId id = hoItr->detid();
           float recal; 
           if (jetRecalib->exists(id))
             recal = jetRecalib->getValues(id)->getValue();
	   else recal = 1.; 
           float energy = hoItr->energy(); 
           float time = hoItr->time();
           HORecHit* hohit = new HORecHit(id,recal*energy,time); 
           miniDiJetsHORecHitCollection->push_back(*hohit);
        }
  } catch (cms::Exception& e) { // can't find it!
  if (!allowMissingInputs_) {cout<<" No HO collection "<<endl; throw e;}
  }

  try {
  edm::Handle<HFRecHitCollection> hf;
  iEvent.getByLabel(hfInput_,hf);
  const HFRecHitCollection Hithf = *(hf.product());
  for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
      {
           DetId id = hfItr->detid();
           float recal; 
           if (jetRecalib->exists(id))
             recal = jetRecalib->getValues(id)->getValue();
	   else recal = 1.; 
           float energy = hfItr->energy(); 
           float time = hfItr->time();
           HFRecHit* hfhit = new HFRecHit(id,recal*energy,time); 
           miniDiJetsHFRecHitCollection->push_back(*hfhit);
      }
  } catch (cms::Exception& e) { // can't find it!
  if (!allowMissingInputs_) throw e;
  }

  //Put selected information in the event

  iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHEReRecHitCollection");

  iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHOReRecHitCollection");

  iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFReRecHitCollection");

}
}

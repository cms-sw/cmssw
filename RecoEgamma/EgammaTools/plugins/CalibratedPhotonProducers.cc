#ifndef RecoEgamma_EgammaTools_CalibratedPhotonProducer_h
#define RecoEgamma_EgammaTools_CalibratedPhotonProducer_h

//author: Alan Smithee
//description: 
//  this class allows the residual scale and smearing to be applied to photon
//  it will write out all the calibration info in the event, such as scale correction value, 
//  smearing correction value, random nr used, energy post calibration, energy pre calibration
//  can optionally write out a new collection of photon with the energy corrected by default
//  a port of EgammaAnalysis/ElectronTools/CalibratedPhotonProducerRun2

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaTools/interface/PhotonEnergyCalibrator.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"
#include "RecoEgamma/EgammaTools/interface/EgammaRandomSeeds.h"

#include "TRandom2.h"

#include <vector>
#include <random>

template<typename T>
class CalibratedPhotonProducerT: public edm::stream::EDProducer<> {
public:
  explicit CalibratedPhotonProducerT( const edm::ParameterSet & ) ;
  ~CalibratedPhotonProducerT() override{};
  void produce( edm::Event &, const edm::EventSetup & ) override ;

private:
  void setSemiDetRandomSeed(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr);

  edm::EDGetTokenT<edm::View<T> > photonToken_;
  PhotonEnergyCalibrator      energyCorrector_;
  std::unique_ptr<TRandom> semiDeterministicRng_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEEToken_;
  bool produceCalibratedObjs_;

  static const std::vector<std::pair<size_t,std::string> > valMapsToStore_;

  typedef edm::ValueMap<float>                     floatMap;
  typedef edm::View<T>                             objectCollection;
  typedef std::vector<T>                           objectVector;
};

template<typename T>
const std::vector<std::pair<size_t,std::string> > CalibratedPhotonProducerT<T>::valMapsToStore_ = {
  {EGEnergySysIndex::kScaleUp,"EGMscaleUpUncertainty"},
  {EGEnergySysIndex::kScaleDown,"EGMscaleDownUncertainty"},
  {EGEnergySysIndex::kScaleStatUp,"EGMscaleStatUpUncertainty"},
  {EGEnergySysIndex::kScaleStatDown,"EGMscaleStatDownUncertainty"},
  {EGEnergySysIndex::kScaleSystUp,"EGMscaleSystUpUncertainty"},
  {EGEnergySysIndex::kScaleSystDown,"EGMscaleSystDownUncertainty"},
  {EGEnergySysIndex::kScaleGainUp,"EGMscaleGainUpUncertainty"},
  {EGEnergySysIndex::kScaleGainDown,"EGMscaleGainDownUncertainty"},
  {EGEnergySysIndex::kSmearUp,"EGMresolutionUpUncertainty"},
  {EGEnergySysIndex::kSmearDown,"EGMresolutionDownUncertainty"},
  {EGEnergySysIndex::kSmearRhoUp,"EGMresolutionRhoUpUncertainty"},
  {EGEnergySysIndex::kSmearRhoDown,"EGMresolutionRhoDownUncertainty"},
  {EGEnergySysIndex::kSmearPhiUp,"EGMresolutionPhiUpUncertainty"},
  {EGEnergySysIndex::kSmearPhiDown,"EGMresolutionPhiDownUncertainty"},
  {EGEnergySysIndex::kScaleValue,"EGMscale"},
  {EGEnergySysIndex::kSmearValue,"EGMsmear"},
  {EGEnergySysIndex::kSmearNrSigma,"EGMsmearNrSigma"},
  {EGEnergySysIndex::kEcalPreCorr,"EGMecalEnergyPreCorr"},
  {EGEnergySysIndex::kEcalErrPreCorr,"EGMecalEnergyErrPreCorr"}, 
  {EGEnergySysIndex::kEcalPostCorr,"EGMecalEnergy"},
  {EGEnergySysIndex::kEcalErrPostCorr,"EGMecalEnergyErr"},
};

namespace{
  template<typename HandleType,typename ValType>
    void fillAndStoreValueMap(edm::Event& iEvent,HandleType objHandle,
			      std::vector<ValType> vals,std::string name)
  {
    auto valMap = std::make_unique<edm::ValueMap<ValType> >();
    typename edm::ValueMap<ValType>::Filler filler(*valMap);
    filler.insert(objHandle,vals.begin(),vals.end());
    filler.fill();
    iEvent.put(std::move(valMap),std::move(name));
  }
}

template<typename T>
CalibratedPhotonProducerT<T>::CalibratedPhotonProducerT( const edm::ParameterSet & conf ) :
  photonToken_(consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("src"))),
  energyCorrector_(conf.getParameter<std::string >("correctionFile")),
  recHitCollectionEBToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEB"))),
  recHitCollectionEEToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEE"))),
  produceCalibratedObjs_(conf.getParameter<bool>("produceCalibratedObjs"))
{ 

  energyCorrector_.setMinEt(conf.getParameter<double>("minEtToCalibrate"));

  if (conf.getParameter<bool>("semiDeterministic")) {
     semiDeterministicRng_.reset(new TRandom2());
     energyCorrector_.initPrivateRng(semiDeterministicRng_.get());
  }

  if(produceCalibratedObjs_) produces<objectVector>();
  
  for(const auto& toStore : valMapsToStore_){
    produces<floatMap>(toStore.second);
  }
}

template<typename T>
void
CalibratedPhotonProducerT<T>::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) {

  edm::Handle<objectCollection> inHandle;
  iEvent.getByToken(photonToken_, inHandle);
  
  edm::Handle<EcalRecHitCollection> recHitCollectionEBHandle;
  edm::Handle<EcalRecHitCollection> recHitCollectionEEHandle;
  
  iEvent.getByToken(recHitCollectionEBToken_, recHitCollectionEBHandle);
  iEvent.getByToken(recHitCollectionEEToken_, recHitCollectionEEHandle);

  std::unique_ptr<objectVector> out = std::make_unique<objectVector>();

  size_t nrObj = inHandle->size();
  std::vector<std::vector<float> > results(EGEnergySysIndex::kNrSysErrs);
  for(auto& res : results) res.reserve(nrObj);

  const PhotonEnergyCalibrator::EventType evtType = iEvent.isRealData() ? 
    PhotonEnergyCalibrator::EventType::DATA : PhotonEnergyCalibrator::EventType::MC; 
  
  for (auto &pho : *inHandle) {
    out->emplace_back(pho);

    if(semiDeterministicRng_) setSemiDetRandomSeed(iEvent,pho,nrObj,out->size());

    const EcalRecHitCollection* recHits = (pho.isEB()) ? recHitCollectionEBHandle.product() : recHitCollectionEEHandle.product();    
    std::vector<float> uncertainties = energyCorrector_.calibrate(out->back(), iEvent.id().run(), recHits, iEvent.streamID(), evtType);
    
    for(size_t index=0;index<EGEnergySysIndex::kNrSysErrs;index++){
      results[index].push_back(uncertainties[index]);
    }
  }
    
  if(produceCalibratedObjs_){
    edm::OrphanHandle<objectVector> outHandle = iEvent.put(std::move(out));
    for(const auto& mapToStore : valMapsToStore_){
      fillAndStoreValueMap(iEvent,outHandle,results[mapToStore.first],mapToStore.second);
    }
  }else{
    for(const auto& mapToStore : valMapsToStore_){
      fillAndStoreValueMap(iEvent,inHandle,results[mapToStore.first],mapToStore.second);
    }
  }

}

//needs to be synced to CalibratedElectronProducers, want the same seed for a given SC
template<typename T>
void CalibratedPhotonProducerT<T>::setSemiDetRandomSeed(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr)
{
  if(obj.superCluster().isNonnull()){
    semiDeterministicRng_->SetSeed(egamma::getRandomSeedFromSC(iEvent,obj.superCluster()));
  }else{
    semiDeterministicRng_->SetSeed(egamma::getRandomSeedFromObj(iEvent,obj,nrObjs,objNr));
  }
}

using CalibratedPhotonProducer = CalibratedPhotonProducerT<reco::Photon>;
using CalibratedPatPhotonProducer = CalibratedPhotonProducerT<pat::Photon>;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedPhotonProducer);
DEFINE_FWK_MODULE(CalibratedPatPhotonProducer);

#endif

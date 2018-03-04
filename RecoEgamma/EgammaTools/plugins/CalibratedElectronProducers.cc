#ifndef RecoEgamma_EgammaTools_CalibratedElectronProducer_h
#define RecoEgamma_EgammaTools_CalibratedElectronProducer_h

//author: Alan Smithee
//description: 
//  this class allows the residual scale and smearing to be applied to electrons
//  the scale and smearing is on the ecal part of the energy
//  hence the E/p combination needs to be re-don, hence the E/p Combination Tools
//  it re-applies the regression with the new corrected ecal energy
//  returns a vector of calibrated energies and correction data, indexed by EGEnergySysIndex
//  a port of EgammaAnalysis/ElectronTools/CalibratedElectronProducerRun2

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"
#include "RecoEgamma/EgammaTools/interface/ElectronEnergyCalibrator.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"
#include "RecoEgamma/EgammaTools/interface/EgammaRandomSeeds.h"

#include "TRandom2.h"

#include <vector>


template<typename T>
class CalibratedElectronProducerT: public edm::stream::EDProducer<>
{
public:
  explicit CalibratedElectronProducerT( const edm::ParameterSet & ) ;
  ~CalibratedElectronProducerT() override{}
  void produce( edm::Event &, const edm::EventSetup & ) override ;
  
private: 
  void setSemiDetRandomSeed(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr);

  edm::EDGetTokenT<edm::View<T> > electronToken_;
  
  EpCombinationTool        epCombinationTool_;
  ElectronEnergyCalibrator energyCorrector_;
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
const std::vector<std::pair<size_t,std::string> > CalibratedElectronProducerT<T>::valMapsToStore_ = {
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
  {EGEnergySysIndex::kEcalTrkPreCorr,"EGMecalTrkEnergyPreCorr"},
  {EGEnergySysIndex::kEcalTrkErrPreCorr,"EGMecalTrkEnergyErrPreCorr"}, 
  {EGEnergySysIndex::kEcalTrkPostCorr,"EGMecalTrkEnergy"},
  {EGEnergySysIndex::kEcalTrkErrPostCorr,"EGMecalTrkEnergyErr"}
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
CalibratedElectronProducerT<T>::CalibratedElectronProducerT( const edm::ParameterSet & conf ) :
  electronToken_(consumes<objectCollection>(conf.getParameter<edm::InputTag>("src"))),
  epCombinationTool_(conf.getParameter<edm::ParameterSet>("epCombConfig")),
  energyCorrector_(epCombinationTool_, conf.getParameter<std::string>("correctionFile")),
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
CalibratedElectronProducerT<T>::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) 
{
  
  epCombinationTool_.setEventContent(iSetup);
  
  edm::Handle<objectCollection> inHandle;
  iEvent.getByToken(electronToken_, inHandle);
  
  edm::Handle<EcalRecHitCollection> recHitCollectionEBHandle;
  edm::Handle<EcalRecHitCollection> recHitCollectionEEHandle;
  
  iEvent.getByToken(recHitCollectionEBToken_, recHitCollectionEBHandle);
  iEvent.getByToken(recHitCollectionEEToken_, recHitCollectionEEHandle);
  
  std::unique_ptr<objectVector> out = std::make_unique<objectVector>();

  size_t nrObj = inHandle->size();
  std::vector<std::vector<float> > results(EGEnergySysIndex::kNrSysErrs);
  for(auto& res : results) res.reserve(nrObj);

  const ElectronEnergyCalibrator::EventType evtType = iEvent.isRealData() ? 
    ElectronEnergyCalibrator::EventType::DATA : ElectronEnergyCalibrator::EventType::MC; 
  
  
  for (auto &ele : *inHandle) {
    out->push_back(ele);
    
    if(semiDeterministicRng_) setSemiDetRandomSeed(iEvent,ele,nrObj,out->size());

    const EcalRecHitCollection* recHits = (ele.isEB()) ? recHitCollectionEBHandle.product() : recHitCollectionEEHandle.product();
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

template<typename T>
void CalibratedElectronProducerT<T>::setSemiDetRandomSeed(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr)
{
  if(obj.superCluster().isNonnull()){
    semiDeterministicRng_->SetSeed(egamma::getRandomSeedFromSC(iEvent,obj.superCluster()));
  }else{
    semiDeterministicRng_->SetSeed(egamma::getRandomSeedFromObj(iEvent,obj,nrObjs,objNr));
  }
}

using CalibratedElectronProducer = CalibratedElectronProducerT<reco::GsfElectron>;
using CalibratedPatElectronProducer = CalibratedElectronProducerT<pat::Electron>; 

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedElectronProducer);
DEFINE_FWK_MODULE(CalibratedPatElectronProducer);

#endif

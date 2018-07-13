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
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
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

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "TRandom2.h"

#include <vector>


template<typename T>
class CalibratedElectronProducerT: public edm::stream::EDProducer<>
{
public:
  explicit CalibratedElectronProducerT( const edm::ParameterSet & ) ;
  ~CalibratedElectronProducerT() override{}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
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
  static const std::vector<int> valMapsToStore_;

};

template<typename T>
const std::vector<int> CalibratedElectronProducerT<T>::valMapsToStore_ = {
  EGEnergySysIndex::kScaleStatUp,
  EGEnergySysIndex::kScaleStatDown,
  EGEnergySysIndex::kScaleSystUp,
  EGEnergySysIndex::kScaleSystDown,
  EGEnergySysIndex::kScaleGainUp,
  EGEnergySysIndex::kScaleGainDown,
  EGEnergySysIndex::kSmearRhoUp,
  EGEnergySysIndex::kSmearRhoDown,
  EGEnergySysIndex::kSmearPhiUp,
  EGEnergySysIndex::kSmearPhiDown,
  EGEnergySysIndex::kScaleUp,
  EGEnergySysIndex::kScaleDown,
  EGEnergySysIndex::kSmearUp,
  EGEnergySysIndex::kSmearDown,
  EGEnergySysIndex::kScaleValue,
  EGEnergySysIndex::kSmearValue,
  EGEnergySysIndex::kSmearNrSigma,
  EGEnergySysIndex::kEcalPreCorr,
  EGEnergySysIndex::kEcalErrPreCorr,
  EGEnergySysIndex::kEcalPostCorr,
  EGEnergySysIndex::kEcalErrPostCorr,
  EGEnergySysIndex::kEcalTrkPreCorr,
  EGEnergySysIndex::kEcalTrkErrPreCorr,
  EGEnergySysIndex::kEcalTrkPostCorr,
  EGEnergySysIndex::kEcalTrkErrPostCorr
};

namespace{
  template<typename HandleType,typename ValType>
    void fillAndStoreValueMap(edm::Event& iEvent,HandleType objHandle,
			      const std::vector<ValType>& vals,const std::string& name)
  {
    auto valMap = std::make_unique<edm::ValueMap<ValType> >();
    typename edm::ValueMap<ValType>::Filler filler(*valMap);
    filler.insert(objHandle,vals.begin(),vals.end());
    filler.fill();
    iEvent.put(std::move(valMap),name);
  }
}

template<typename T>
CalibratedElectronProducerT<T>::CalibratedElectronProducerT( const edm::ParameterSet & conf ) :
  electronToken_(consumes<edm::View<T>>(conf.getParameter<edm::InputTag>("src"))),
  epCombinationTool_(conf.getParameter<edm::ParameterSet>("epCombConfig")),
  energyCorrector_(epCombinationTool_, conf.getParameter<std::string>("correctionFile")),
  recHitCollectionEBToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEB"))),
  recHitCollectionEEToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEE"))),
  produceCalibratedObjs_(conf.getParameter<bool>("produceCalibratedObjs"))
{
  energyCorrector_.setMinEt(conf.getParameter<double>("minEtToCalibrate"));  
  energyCorrector_.setUseSmearCorrEcalEnergyErrInComb(conf.getParameter<bool>("useSmearCorrEcalEnergyErrInComb"));  
  
  if (conf.getParameter<bool>("semiDeterministic")) {
     semiDeterministicRng_.reset(new TRandom2());
     energyCorrector_.initPrivateRng(semiDeterministicRng_.get());
  }

  if(produceCalibratedObjs_) produces<std::vector<T>>();
  
  for(const auto& toStore : valMapsToStore_){
    produces<edm::ValueMap<float>>(EGEnergySysIndex::name(toStore));
  }
}

template<typename T>
void CalibratedElectronProducerT<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src",edm::InputTag("gedPhotons"));
  desc.add<edm::ParameterSetDescription>("epCombConfig",EpCombinationTool::makePSetDescription());
  desc.add<edm::InputTag>("recHitCollectionEB",edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("recHitCollectionEE",edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<std::string>("correctionFile",std::string());
  desc.add<double>("minEtToCalibrate",5.0);
  desc.add<bool>("produceCalibratedObjs",true);
  desc.add<bool>("semiDeterministic",true);
  desc.add<bool>("useSmearCorrEcalEnergyErrInComb",false);
  std::vector<std::string> valMapsProduced;
  for(auto varToStore : valMapsToStore_) valMapsProduced.push_back(EGEnergySysIndex::name(varToStore));
  desc.add<std::vector<std::string> >("valueMapsStored",valMapsProduced)->setComment("provides to python configs the list of valuemaps stored, can not be overriden in the python config");
  descriptions.add(defaultModuleLabel<CalibratedElectronProducerT<T>>(),desc);
}

template<typename T>
void
CalibratedElectronProducerT<T>::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) 
{
  
  epCombinationTool_.setEventContent(iSetup);
  
  edm::Handle<edm::View<T>> inHandle;
  iEvent.getByToken(electronToken_, inHandle);
  
  edm::Handle<EcalRecHitCollection> recHitCollectionEBHandle;
  edm::Handle<EcalRecHitCollection> recHitCollectionEEHandle;
  
  iEvent.getByToken(recHitCollectionEBToken_, recHitCollectionEBHandle);
  iEvent.getByToken(recHitCollectionEEToken_, recHitCollectionEEHandle);
  
  std::unique_ptr<std::vector<T>> out = std::make_unique<std::vector<T>>();

  size_t nrObj = inHandle->size();
  std::array<std::vector<float>,EGEnergySysIndex::kNrSysErrs> results;
  for(auto& res : results) res.reserve(nrObj);

  const ElectronEnergyCalibrator::EventType evtType = iEvent.isRealData() ? 
    ElectronEnergyCalibrator::EventType::DATA : ElectronEnergyCalibrator::EventType::MC; 
  
  
  for (const auto& ele : *inHandle) {
    out->push_back(ele);
    
    if(semiDeterministicRng_) setSemiDetRandomSeed(iEvent,ele,nrObj,out->size());

    const EcalRecHitCollection* recHits = (ele.isEB()) ? recHitCollectionEBHandle.product() : recHitCollectionEEHandle.product();
    std::array<float,EGEnergySysIndex::kNrSysErrs> uncertainties = energyCorrector_.calibrate(out->back(), iEvent.id().run(), recHits, iEvent.streamID(), evtType);
    
    for(size_t index=0;index<EGEnergySysIndex::kNrSysErrs;index++){
      results[index].push_back(uncertainties[index]);
    }
  }

  auto fillAndStore = [&](auto handle){
    for(const auto& mapToStore : valMapsToStore_){
      fillAndStoreValueMap(iEvent,handle,results[mapToStore],EGEnergySysIndex::name(mapToStore));
    }
  };
  
  if(produceCalibratedObjs_){
    fillAndStore(iEvent.put(std::move(out)));
  }else{
    fillAndStore(inHandle);
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

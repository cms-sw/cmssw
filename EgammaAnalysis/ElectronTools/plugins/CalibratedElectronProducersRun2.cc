#ifndef CalibratedElectronProducerRun2_h
#define CalibratedElectronProducerRun2_h

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
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationToolSemi.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibratorRun2.h"
#include "EgammaAnalysis/ElectronTools/interface/EGEnergySysIndex.h"

#include "TRandom2.h"

#include <vector>
#include <random>

template<typename T>
class CalibratedElectronProducerRun2T: public edm::stream::EDProducer<>
{
public:
  explicit CalibratedElectronProducerRun2T( const edm::ParameterSet & ) ;
  ~CalibratedElectronProducerRun2T() override;
  void produce( edm::Event &, const edm::EventSetup & ) override ;
  
private: 
  void setSemiDetRandomSeed(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr);

  edm::EDGetTokenT<edm::View<T> > theElectronToken;
  std::vector<std::string>        theGBRForestName;
  std::vector<const GBRForestD* > theGBRForestHandle;
  
  EpCombinationToolSemi        theEpCombinationTool;
  ElectronEnergyCalibratorRun2 theEnCorrectorRun2;
  std::unique_ptr<TRandom> theSemiDeterministicRng;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEEToken_;
  bool autoDataType_;
  bool produceCalibratedEles_;
  static const std::vector<std::pair<size_t,std::string> > valMapsToStore_;

  typedef edm::ValueMap<float>                     floatMap;
  typedef edm::View<T>                             objectCollection;
  typedef std::vector<T>                           objectVector;

};

template<typename T>
const std::vector<std::pair<size_t,std::string> > CalibratedElectronProducerRun2T<T>::valMapsToStore_ = {
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
CalibratedElectronProducerRun2T<T>::CalibratedElectronProducerRun2T( const edm::ParameterSet & conf ) :
  theElectronToken(consumes<objectCollection>(conf.getParameter<edm::InputTag>("electrons"))),
  theGBRForestName(conf.getParameter< std::vector<std::string> >("gbrForestName")),
  theEpCombinationTool(),
  theEnCorrectorRun2(theEpCombinationTool, conf.getParameter<bool>("isMC"), conf.getParameter<bool>("isSynchronization"), conf.getParameter<std::string>("correctionFile")),
  recHitCollectionEBToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEB"))),
  recHitCollectionEEToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEE"))),
  autoDataType_((conf.existsAs<bool>("autoDataType") && !conf.getParameter<bool>("autoDataType") ) ? false : true),
  produceCalibratedEles_(conf.getParameter<bool>("produceCalibratedEles"))
{
  theEnCorrectorRun2.setMinEt(conf.getParameter<double>("minEtToCalibrate"));  
  
  if (conf.existsAs<bool>("semiDeterministic") && conf.getParameter<bool>("semiDeterministic")) {
     theSemiDeterministicRng.reset(new TRandom2());
     theEnCorrectorRun2.initPrivateRng(theSemiDeterministicRng.get());
  }

  if(produceCalibratedEles_) produces<objectVector>();
  
  for(const auto& toStore : valMapsToStore_){
    produces<floatMap>(toStore.second);
  }
}

template<typename T>
CalibratedElectronProducerRun2T<T>::~CalibratedElectronProducerRun2T()
{
}

template<typename T>
void
CalibratedElectronProducerRun2T<T>::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) 
{
  
  for (auto&& forestName : theGBRForestName) {
    edm::ESHandle<GBRForestD> forestHandle;
    iSetup.get<GBRDWrapperRcd>().get(forestName, forestHandle);
    theGBRForestHandle.push_back(forestHandle.product());      
  }
  
  theEpCombinationTool.init(theGBRForestHandle);
  
  edm::Handle<objectCollection> inHandle;
  iEvent.getByToken(theElectronToken, inHandle);
  
  edm::Handle<EcalRecHitCollection> recHitCollectionEBHandle;
  edm::Handle<EcalRecHitCollection> recHitCollectionEEHandle;
  
  iEvent.getByToken(recHitCollectionEBToken_, recHitCollectionEBHandle);
  iEvent.getByToken(recHitCollectionEEToken_, recHitCollectionEEHandle);
  
  std::unique_ptr<objectVector> out = std::make_unique<objectVector>();

  size_t nrObj = inHandle->size();
  std::vector<std::vector<float> > results(EGEnergySysIndex::kNrSysErrs);
  for(auto& res : results) res.reserve(nrObj);

  int eventIsMC = -1;
  if (autoDataType_) eventIsMC = iEvent.isRealData() ? 0 : 1;
  
  for (auto &ele : *inHandle) {
    out->push_back(ele);
    
    if(theSemiDeterministicRng) setSemiDetRandomSeed(iEvent,ele,nrObj,out->size());

    const EcalRecHitCollection* recHits = (ele.isEB()) ? recHitCollectionEBHandle.product() : recHitCollectionEEHandle.product();
    std::vector<float> uncertainties = theEnCorrectorRun2.calibrate(out->back(), iEvent.id().run(), recHits, iEvent.streamID(), eventIsMC);
    
    for(size_t index=0;index<EGEnergySysIndex::kNrSysErrs;index++){
      results[index].push_back(uncertainties[index]);
    }
  }
  if(produceCalibratedEles_){
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

//needs to be synced to CalibratedPhotonProducersRun2, want the same seed for a given SC
template<typename T>
void CalibratedElectronProducerRun2T<T>::setSemiDetRandomSeed(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr)
{
  if(obj.superCluster().isNonnull()){
    auto scRef = obj.superCluster();
    std::seed_seq seeder = {int(iEvent.id().event()), int(iEvent.id().luminosityBlock()), int(iEvent.id().run()),
			    int(scRef->seed()->seed().rawId()),int(scRef->hitsAndFractions().size())};
    uint32_t seed = 0, tries = 10;
    do {
      seeder.generate(&seed,&seed+1); tries++;
    } while (seed == 0 && tries < 10);
    theSemiDeterministicRng->SetSeed(seed ? seed : iEvent.id().event() + 10000*scRef.key() );
  }else{
     std::seed_seq seeder = {int(iEvent.id().event()), int(iEvent.id().luminosityBlock()), int(iEvent.id().run()),
			     int(nrObjs),int(std::numeric_limits<int>::max()*obj.phi()/M_PI) & 0xFFF,int(objNr)};	   
     uint32_t seed = 0, tries = 10;
     do {
       seeder.generate(&seed,&seed+1); tries++;
     } while (seed == 0 && tries < 10);
     theSemiDeterministicRng->SetSeed(seed ? seed : iEvent.id().event() + 10000*objNr );
  }
}

typedef CalibratedElectronProducerRun2T<reco::GsfElectron> CalibratedElectronProducerRun2;
typedef CalibratedElectronProducerRun2T<pat::Electron> CalibratedPatElectronProducerRun2;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedElectronProducerRun2);
DEFINE_FWK_MODULE(CalibratedPatElectronProducerRun2);

#endif

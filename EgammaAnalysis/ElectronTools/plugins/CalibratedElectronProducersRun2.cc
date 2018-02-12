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

#include <vector>
#include <memory>

template<typename T>
class CalibratedElectronProducerRun2T: public edm::stream::EDProducer<>
{
public:
  explicit CalibratedElectronProducerRun2T( const edm::ParameterSet & ) ;
  virtual ~CalibratedElectronProducerRun2T();
  virtual void produce( edm::Event &, const edm::EventSetup & ) override ;
  
private:
  edm::EDGetTokenT<edm::View<T> > theElectronToken;
  std::vector<std::string>        theGBRForestName;
  std::vector<const GBRForestD* > theGBRForestHandle;
  
  EpCombinationToolSemi        theEpCombinationTool;
  ElectronEnergyCalibratorRun2 theEnCorrectorRun2;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEEToken_;
  bool autoDataType;

  typedef edm::ValueMap<float>                     floatMap;
  typedef edm::View<T>                             objectCollection;
  typedef std::vector<T>                           objectVector;

};

template<typename T>
CalibratedElectronProducerRun2T<T>::CalibratedElectronProducerRun2T( const edm::ParameterSet & conf ) :
  theElectronToken(consumes<objectCollection>(conf.getParameter<edm::InputTag>("electrons"))),
  theGBRForestName(conf.getParameter< std::vector<std::string> >("gbrForestName")),
  theEpCombinationTool(),
  theEnCorrectorRun2(theEpCombinationTool, conf.getParameter<bool>("isMC"), conf.getParameter<bool>("isSynchronization"), conf.getParameter<std::string>("correctionFile")),
  recHitCollectionEBToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEB"))),
  recHitCollectionEEToken_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("recHitCollectionEE"))),
  autoDataType((conf.existsAs<bool>("autoDataType") && !conf.getParameter<bool>("autoDataType") ) ? 0 : 1)
{
  produces<objectVector>();
  produces<floatMap>("EGMscaleStatUpUncertainty");
  produces<floatMap>("EGMscaleStatDownUncertainty");
  produces<floatMap>("EGMscaleSystUpUncertainty");
  produces<floatMap>("EGMscaleSystDownUncertainty");
  produces<floatMap>("EGMscaleGainUpUncertainty");
  produces<floatMap>("EGMscaleGainDownUncertainty");
  produces<floatMap>("EGMresolutionRhoUpUncertainty");
  produces<floatMap>("EGMresolutionRhoDownUncertainty");
  produces<floatMap>("EGMresolutionPhiUpUncertainty");
  produces<floatMap>("EGMresolutionPhiDownUncertainty");

  produces<floatMap>("EGMscaleUpUncertainty");
  produces<floatMap>("EGMscaleDownUncertainty");
  produces<floatMap>("EGMresolutionUpUncertainty");
  produces<floatMap>("EGMresolutionDownUncertainty");

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
  
  edm::Handle<objectCollection> in;
  iEvent.getByToken(theElectronToken, in);
  
  edm::Handle<EcalRecHitCollection> recHitCollectionEBHandle;
  edm::Handle<EcalRecHitCollection> recHitCollectionEEHandle;
  
  iEvent.getByToken(recHitCollectionEBToken_, recHitCollectionEBHandle);
  iEvent.getByToken(recHitCollectionEEToken_, recHitCollectionEEHandle);
  
  std::unique_ptr<objectVector> out = std::make_unique<objectVector>();
  
  std::vector<float> stat_up;
  std::vector<float> stat_dn;
  std::vector<float> syst_up;
  std::vector<float> syst_dn;
  std::vector<float> gain_up;
  std::vector<float> gain_dn;
  std::vector<float> rho_up;
  std::vector<float> rho_dn;
  std::vector<float> phi_up;
  std::vector<float> phi_dn;

  std::vector<float> scale_up;
  std::vector<float> scale_dn;
  std::vector<float> resol_up;
  std::vector<float> resol_dn;

  out->reserve(in->size());   

  stat_up.reserve(in->size());   
  stat_dn.reserve(in->size());   
  syst_up.reserve(in->size());   
  syst_dn.reserve(in->size());   
  gain_up.reserve(in->size());   
  gain_dn.reserve(in->size());   
  rho_up.reserve(in->size());   
  rho_dn.reserve(in->size());   
  phi_up.reserve(in->size());   
  phi_dn.reserve(in->size());   

  scale_up.reserve(in->size());
  scale_dn.reserve(in->size());
  resol_up.reserve(in->size());
  resol_dn.reserve(in->size());
  
  int eventIsMC = -1;
  if (autoDataType)
    eventIsMC = iEvent.isRealData() ? 0 : 1;
  
  for (auto &ele : *in) {
    out->push_back(ele);
    const EcalRecHitCollection* recHits = (ele.isEB()) ? recHitCollectionEBHandle.product() : recHitCollectionEEHandle.product();
    std::vector<float> uncertainties = theEnCorrectorRun2.calibrate(out->back(), iEvent.id().run(), recHits, iEvent.streamID(), eventIsMC);

    stat_up.push_back(uncertainties[0]);
    stat_dn.push_back(uncertainties[1]);
    syst_up.push_back(uncertainties[2]);
    syst_dn.push_back(uncertainties[3]);
    gain_up.push_back(uncertainties[4]);
    gain_dn.push_back(uncertainties[5]);
    rho_up.push_back (uncertainties[6]);
    rho_dn.push_back (uncertainties[7]);
    phi_up.push_back (uncertainties[8]);
    phi_dn.push_back (uncertainties[9]);

    scale_up.push_back(uncertainties[10]);
    scale_dn.push_back(uncertainties[11]);
    resol_up.push_back(uncertainties[12]);
    resol_dn.push_back(uncertainties[13]);
  }
  
  std::unique_ptr<floatMap> statUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> statDownMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> systUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> systDownMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> gainUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> gainDownMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> rhoUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> rhoDownMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> phiUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> phiDownMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> scaleUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> scaleDownMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> resolUpMap = std::make_unique<floatMap>();
  std::unique_ptr<floatMap> resolDownMap = std::make_unique<floatMap>();

  edm::OrphanHandle<objectVector> calibratedHandle = iEvent.put(std::move(out));

  floatMap::Filler statUpMapFiller(*statUpMap);
  statUpMapFiller.insert(calibratedHandle, stat_up.begin(), stat_up.end());
  statUpMapFiller.fill();
  iEvent.put(std::move(statUpMap), "EGMscaleStatUpUncertainty");

  floatMap::Filler statDownMapFiller(*statDownMap);
  statDownMapFiller.insert(calibratedHandle, stat_dn.begin(), stat_dn.end());
  statDownMapFiller.fill();
  iEvent.put(std::move(statDownMap), "EGMscaleStatDownUncertainty");

  floatMap::Filler systUpMapFiller(*systUpMap);
  systUpMapFiller.insert(calibratedHandle, syst_up.begin(), syst_up.end());
  systUpMapFiller.fill();
  iEvent.put(std::move(systUpMap), "EGMscaleSystUpUncertainty");

  floatMap::Filler systDownMapFiller(*systDownMap);
  systDownMapFiller.insert(calibratedHandle, syst_dn.begin(), syst_dn.end());
  systDownMapFiller.fill();
  iEvent.put(std::move(systDownMap), "EGMscaleSystDownUncertainty");

  floatMap::Filler gainUpMapFiller(*gainUpMap);
  gainUpMapFiller.insert(calibratedHandle, gain_up.begin(), gain_up.end());
  gainUpMapFiller.fill();
  iEvent.put(std::move(gainUpMap), "EGMscaleGainUpUncertainty");

  floatMap::Filler gainDownMapFiller(*gainDownMap);
  gainDownMapFiller.insert(calibratedHandle, gain_dn.begin(), gain_dn.end());
  gainDownMapFiller.fill();
  iEvent.put(std::move(gainDownMap), "EGMscaleGainDownUncertainty");

  floatMap::Filler rhoUpMapFiller(*rhoUpMap);
  rhoUpMapFiller.insert(calibratedHandle, rho_up.begin(), rho_up.end());
  rhoUpMapFiller.fill();
  iEvent.put(std::move(rhoUpMap), "EGMresolutionRhoUpUncertainty");

  floatMap::Filler rhoDownMapFiller(*rhoDownMap);
  rhoDownMapFiller.insert(calibratedHandle, rho_dn.begin(), rho_dn.end());
  rhoDownMapFiller.fill();
  iEvent.put(std::move(rhoDownMap), "EGMresolutionRhoDownUncertainty");

  floatMap::Filler phiUpMapFiller(*phiUpMap);
  phiUpMapFiller.insert(calibratedHandle, phi_up.begin(), phi_up.end());
  phiUpMapFiller.fill();
  iEvent.put(std::move(phiUpMap), "EGMresolutionPhiUpUncertainty");

  floatMap::Filler phiDownMapFiller(*phiDownMap);
  phiDownMapFiller.insert(calibratedHandle, phi_dn.begin(), phi_dn.end());
  phiDownMapFiller.fill();
  iEvent.put(std::move(phiDownMap), "EGMresolutionPhiDownUncertainty");

  floatMap::Filler scaleUpMapFiller(*scaleUpMap);
  scaleUpMapFiller.insert(calibratedHandle, scale_up.begin(), scale_up.end());
  scaleUpMapFiller.fill();
  iEvent.put(std::move(scaleUpMap), "EGMscaleUpUncertainty");

  floatMap::Filler scaleDownMapFiller(*scaleDownMap);
  scaleDownMapFiller.insert(calibratedHandle, scale_dn.begin(), scale_dn.end());
  scaleDownMapFiller.fill();
  iEvent.put(std::move(scaleDownMap), "EGMscaleDownUncertainty");

  floatMap::Filler resolUpMapFiller(*resolUpMap);
  resolUpMapFiller.insert(calibratedHandle, resol_up.begin(), resol_up.end());
  resolUpMapFiller.fill();
  iEvent.put(std::move(resolUpMap), "EGMresolutionUpUncertainty");

  floatMap::Filler resolDownMapFiller(*resolDownMap);
  resolDownMapFiller.insert(calibratedHandle, resol_dn.begin(), resol_dn.end());
  resolDownMapFiller.fill();
  iEvent.put(std::move(resolDownMap), "EGMresolutionDownUncertainty");


}

typedef CalibratedElectronProducerRun2T<reco::GsfElectron> CalibratedElectronProducerRun2;
typedef CalibratedElectronProducerRun2T<pat::Electron> CalibratedPatElectronProducerRun2;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedElectronProducerRun2);
DEFINE_FWK_MODULE(CalibratedPatElectronProducerRun2);

#endif

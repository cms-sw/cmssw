#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "RecoEgamma/EgammaTools/interface/MultiToken.h"
#include "RecoEgamma/EgammaTools/interface/Utils.h"

#include <memory>
#include <vector>

class ElectronIDValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronIDValueMapProducer(const edm::ParameterSet&);
  ~ElectronIDValueMapProducer() override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::unique_ptr<noZS::EcalClusterLazyTools> lazyToolnoZS;

  // for AOD and MiniAOD case
  MultiTokenT<edm::View<reco::GsfElectron>> src_;
  MultiTokenT<EcalRecHitCollection> ebReducedRecHitCollection_;
  MultiTokenT<EcalRecHitCollection> eeReducedRecHitCollection_;
  MultiTokenT<EcalRecHitCollection> esReducedRecHitCollection_;

  constexpr static char eleFull5x5SigmaIEtaIEta_[] = "eleFull5x5SigmaIEtaIEta";
  constexpr static char eleFull5x5SigmaIEtaIPhi_[] = "eleFull5x5SigmaIEtaIPhi";
  constexpr static char eleFull5x5E1x5_[] = "eleFull5x5E1x5";
  constexpr static char eleFull5x5E2x5_[] = "eleFull5x5E2x5";
  constexpr static char eleFull5x5E5x5_[] = "eleFull5x5E5x5";
  constexpr static char eleFull5x5R9_[] = "eleFull5x5R9";
  constexpr static char eleFull5x5Circularity_[] = "eleFull5x5Circularity";
};

ElectronIDValueMapProducer::ElectronIDValueMapProducer(const edm::ParameterSet& iConfig)
  // Declare consummables, handle both AOD and miniAOD case
  : src_(
        mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("src")),
        mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD")))
  , ebReducedRecHitCollection_(src_,
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection")),
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollectionMiniAOD")))
  , eeReducedRecHitCollection_(src_,
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection")),
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollectionMiniAOD")))
  , esReducedRecHitCollection_(src_,
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("esReducedRecHitCollection")),
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("esReducedRecHitCollectionMiniAOD")))
{

  produces<edm::ValueMap<float> >(eleFull5x5SigmaIEtaIEta_);  
  produces<edm::ValueMap<float> >(eleFull5x5SigmaIEtaIPhi_); 
  produces<edm::ValueMap<float> >(eleFull5x5E1x5_);
  produces<edm::ValueMap<float> >(eleFull5x5E2x5_);
  produces<edm::ValueMap<float> >(eleFull5x5E5x5_);
  produces<edm::ValueMap<float> >(eleFull5x5R9_);  
  produces<edm::ValueMap<float> >(eleFull5x5Circularity_);  

}

ElectronIDValueMapProducer::~ElectronIDValueMapProducer() {
}

void ElectronIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get handle on electrons
  auto src = src_.getValidHandle(iEvent);

  lazyToolnoZS = std::make_unique<noZS::EcalClusterLazyTools>(iEvent, iSetup, 
                              ebReducedRecHitCollection_.get(iEvent), 
                              eeReducedRecHitCollection_.get(iEvent),
                              esReducedRecHitCollection_.get(iEvent));
 
  // size_t n = src->size();
  std::vector<float> eleFull5x5SigmaIEtaIEta, eleFull5x5SigmaIEtaIPhi;
  std::vector<float> eleFull5x5R9, eleFull5x5Circularity;
  std::vector<float> eleFull5x5E1x5,eleFull5x5E2x5,eleFull5x5E5x5;
  
  // reco::GsfElectron::superCluster() is virtual so we can exploit polymorphism
  for (size_t i = 0; i < src->size(); ++i){
    auto iEle = src->ptrAt(i);
    const auto& theseed = *(iEle->superCluster()->seed());

    std::vector<float> vCov = lazyToolnoZS->localCovariances( theseed );
    const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
    const float sep = vCov[1];
    eleFull5x5SigmaIEtaIEta.push_back(see);
    eleFull5x5SigmaIEtaIPhi.push_back(sep);
    eleFull5x5R9.push_back(lazyToolnoZS->e3x3( theseed ) / iEle->superCluster()->rawEnergy() );    
    
    const float e1x5 = lazyToolnoZS->e1x5( theseed );
    const float e2x5 = lazyToolnoZS->e2x5Max( theseed );
    const float e5x5 = lazyToolnoZS->e5x5( theseed );
    const float circularity = (e5x5 != 0.) ? 1.-e1x5/e5x5 : -1;
    
    eleFull5x5E1x5.push_back(e1x5); 
    eleFull5x5E2x5.push_back(e2x5);
    eleFull5x5E5x5.push_back(e5x5);
    eleFull5x5Circularity.push_back(circularity);
  }
  
  writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIEta, eleFull5x5SigmaIEtaIEta_);  
  writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIPhi, eleFull5x5SigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, eleFull5x5R9, eleFull5x5R9_);  
  writeValueMap(iEvent, src, eleFull5x5E1x5, eleFull5x5E1x5_);  
  writeValueMap(iEvent, src, eleFull5x5E2x5, eleFull5x5E2x5_);   
  writeValueMap(iEvent, src, eleFull5x5E5x5, eleFull5x5E5x5_);  
  writeValueMap(iEvent, src, eleFull5x5Circularity, eleFull5x5Circularity_);  
}

void ElectronIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // electronIDValueMapProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("srcMiniAOD", edm::InputTag("slimmedElectrons","","@skipCurrentProcess"));
  desc.add<edm::InputTag>("ebReducedRecHitCollection", edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("eeReducedRecHitCollection", edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<edm::InputTag>("esReducedRecHitCollection", edm::InputTag("reducedEcalRecHitsES"));
  desc.add<edm::InputTag>("ebReducedRecHitCollectionMiniAOD", edm::InputTag("reducedEgamma","reducedEBRecHits"));
  desc.add<edm::InputTag>("eeReducedRecHitCollectionMiniAOD", edm::InputTag("reducedEgamma","reducedEERecHits"));
  desc.add<edm::InputTag>("esReducedRecHitCollectionMiniAOD", edm::InputTag("reducedEgamma","reducedESRecHits"));
  descriptions.add("electronIDValueMapProducer", desc);
}


DEFINE_FWK_MODULE(ElectronIDValueMapProducer);

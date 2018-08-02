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
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"

#include "RecoEgamma/EgammaTools/interface/Utils.h"
#include "RecoEgamma/EgammaTools/interface/MultiToken.h"

#include <memory>
#include <vector>
#include <unordered_map>

namespace {
  enum reg_float_vars { k_NFloatVars = 0 };
  
  enum reg_int_vars   { k_NIntVars   = 0 };

  const std::vector<std::string> float_var_names( { } );
  
  const std::vector<std::string> integer_var_names( { } );  
  
  inline void set_map_val( const reg_float_vars index, const float value,
                           std::unordered_map<std::string,float>& map) {
    map[float_var_names[index]] = value;
  }
  inline void set_map_val( const reg_int_vars index, const int value,
                           std::unordered_map<std::string,int>& map) {
    map[integer_var_names[index]] = value;
  }

  template<typename T>
  inline void check_map(const std::unordered_map<std::string,T>& map, unsigned int exp_size) {
    if( map.size() != exp_size ) {
      throw cms::Exception("ElectronRegressionWeirdConfig")
        << "variable map size: " << map.size() 
        << " not equal to expected size: " << exp_size << " !"
        << " The regression variable calculation code definitely has a bug, fix it!";
    }
  }

  template<typename LazyTools>
  void calculateValues(EcalClusterLazyToolsBase* tools_tocast,
                       const edm::Ptr<reco::GsfElectron>& iEle,
                       const edm::EventSetup& iSetup,
                       std::unordered_map<std::string,float>& float_vars,
                       std::unordered_map<std::string,int>& int_vars ) {    
  }
}

class ElectronRegressionValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronRegressionValueMapProducer(const edm::ParameterSet&);
  ~ElectronRegressionValueMapProducer() override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  void produce(edm::Event&, const edm::EventSetup&) override;

  const bool use_full5x5_;

  // for AOD and MiniAOD case
  MultiToken<EcalRecHitCollection> ebReducedRecHitCollection_;
  MultiToken<EcalRecHitCollection> eeReducedRecHitCollection_;
  MultiToken<EcalRecHitCollection> esReducedRecHitCollection_;
  MultiToken<edm::View<reco::GsfElectron>> src_;
};

ElectronRegressionValueMapProducer::ElectronRegressionValueMapProducer(const edm::ParameterSet& iConfig)
  : use_full5x5_(iConfig.getParameter<bool>("useFull5x5"))
  // Declare consummables, handle both AOD and miniAOD case
  , ebReducedRecHitCollection_(
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection")),
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollectionMiniAOD")))
  , eeReducedRecHitCollection_(
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection")),
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollectionMiniAOD")))
  , esReducedRecHitCollection_(
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("esReducedRecHitCollection")),
        mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("esReducedRecHitCollectionMiniAOD")))
  , src_(
        mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("src")),
        mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD")))
{

  for( const std::string& name : float_var_names ) {
    produces<edm::ValueMap<float> >(name);
  }

  for( const std::string& name : integer_var_names ) {
    produces<edm::ValueMap<int> >(name);
  }  
}

ElectronRegressionValueMapProducer::~ElectronRegressionValueMapProducer() {
}

void ElectronRegressionValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get handle on electrons
  auto src = src_.getHandle(iEvent);

  // configure lazy tools
  edm::EDGetTokenT<EcalRecHitCollection> ebrh = ebReducedRecHitCollection_.get(iEvent);
  edm::EDGetTokenT<EcalRecHitCollection> eerh = eeReducedRecHitCollection_.get(iEvent);
  edm::EDGetTokenT<EcalRecHitCollection> esrh = esReducedRecHitCollection_.get(iEvent);
  
  std::unique_ptr<EcalClusterLazyToolsBase> lazyTools;
  if( use_full5x5_ ) lazyTools =
      std::make_unique<noZS::EcalClusterLazyTools>( iEvent, iSetup, ebrh, eerh, esrh );
  else lazyTools = std::make_unique<EcalClusterLazyTools>( iEvent, iSetup, ebrh, eerh, esrh );

  std::vector<std::vector<float> > float_vars(k_NFloatVars);
  std::vector<std::vector<int> > int_vars(k_NIntVars);
  
  std::unordered_map<std::string,float> float_vars_map;
  std::unordered_map<std::string,int> int_vars_map;

  // reco::GsfElectron::superCluster() is virtual so we can exploit polymorphism
  for (size_t i = 0; i < src->size(); ++i){
    auto iEle = src->ptrAt(i);

    if( use_full5x5_ ) {
      calculateValues<noZS::EcalClusterLazyTools>(lazyTools.get(),
                                                  iEle,
                                                  iSetup,
                                                  float_vars_map,
                                                  int_vars_map);
    } else {
      calculateValues<EcalClusterLazyTools>(lazyTools.get(),
                                            iEle,
                                            iSetup,
                                            float_vars_map,
                                            int_vars_map);
    }

    check_map(float_vars_map, k_NFloatVars);
    check_map(int_vars_map, k_NIntVars);
    
    for( unsigned int i = 0; i < float_vars.size(); ++i ) {
      float_vars[i].emplace_back(float_vars_map.at(float_var_names[i]));
    }

    for( unsigned int i = 0; i < int_vars.size(); ++i ) {
      int_vars[i].emplace_back(int_vars_map.at(integer_var_names[i]));
    }
  }
  
  for( unsigned int i = 0; i < float_vars.size(); ++i ) {
    writeValueMap(iEvent, src, float_vars[i], float_var_names[i]);
  }
  
  for( unsigned int i = 0; i < int_vars.size(); ++i ) {
    writeValueMap(iEvent, src, int_vars[i], integer_var_names[i]);
  }
  
  lazyTools.reset();
}

void ElectronRegressionValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronRegressionValueMapProducer);

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "RecoEgamma/EgammaTools/interface/Utils.h"
#include "RecoEgamma/EgammaTools/interface/MultiToken.h"

#include <memory>
#include <vector>
#include <unordered_map>

namespace {
  // Cluster shapes
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
      throw cms::Exception("PhotonRegressionWeirdConfig")
        << "variable map size: " << map.size() 
        << " not equal to expected size: " << exp_size << " !"
        << " The regression variable calculation code definitely has a bug, fix it!";
    }
  }

  template<typename LazyTools,typename SeedType>
  inline void calculateValues(EcalClusterLazyToolsBase* tools_tocast,
                              const SeedType& the_seed,
                              std::unordered_map<std::string,float>& float_vars,
                              std::unordered_map<std::string,int>& int_vars ) {
  }
}

class PhotonRegressionValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit PhotonRegressionValueMapProducer(const edm::ParameterSet&);
  ~PhotonRegressionValueMapProducer() override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
  
  void produce(edm::Event&, const edm::EventSetup&) override;

  const bool use_full5x5_;

  // for AOD and MiniAOD case
  MultiTokenT<edm::View<reco::Photon>> src_;
  MultiTokenT<EcalRecHitCollection>    ebRecHits_;
  MultiTokenT<EcalRecHitCollection>    eeRecHits_;
  MultiTokenT<EcalRecHitCollection>    esRecHits_;
};

PhotonRegressionValueMapProducer::PhotonRegressionValueMapProducer(const edm::ParameterSet& iConfig)
  : use_full5x5_(iConfig.getParameter<bool>("useFull5x5"))
  // Declare consummables, handle both AOD and miniAOD case
  , src_(            consumesCollector(), iConfig, "src", "srcMiniAOD")
  , ebRecHits_(src_, consumesCollector(), iConfig, "ebReducedRecHitCollection", "ebReducedRecHitCollectionMiniAOD")
  , eeRecHits_(src_, consumesCollector(), iConfig, "eeReducedRecHitCollection", "eeReducedRecHitCollectionMiniAOD")
  , esRecHits_(src_, consumesCollector(), iConfig, "esReducedRecHitCollection", "esReducedRecHitCollectionMiniAOD")
{
  //
  // Declare producibles
  //
  // Cluster shapes
  for( const std::string& name : float_var_names ) {
    produces<edm::ValueMap<float> >(name);
  }

  for( const std::string& name : integer_var_names ) {
    produces<edm::ValueMap<int> >(name);
  }  
}

PhotonRegressionValueMapProducer::~PhotonRegressionValueMapProducer() 
{}

void PhotonRegressionValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get handle on photons
  auto src = src_.getValidHandle(iEvent);
  bool isAOD = src_.getGoodTokenIndex() == 0;

  // Configure lazy tools, which will compute 5x5 quantities  
  edm::EDGetTokenT<EcalRecHitCollection> ebrh = ebRecHits_.get(iEvent);
  edm::EDGetTokenT<EcalRecHitCollection> eerh = eeRecHits_.get(iEvent);
  edm::EDGetTokenT<EcalRecHitCollection> esrh = esRecHits_.get(iEvent);
  
  std::unique_ptr<EcalClusterLazyToolsBase> lazyTools;
  if( use_full5x5_ ) lazyTools =
      std::make_unique<noZS::EcalClusterLazyTools>( iEvent, iSetup, ebrh, eerh, esrh );
  else lazyTools = std::make_unique<EcalClusterLazyTools>( iEvent, iSetup, ebrh, eerh, esrh );

  if( !isAOD && !src->empty() ) {
    edm::Ptr<pat::Photon> test(src->ptrAt(0));
    if( test.isNull() || !test.isAvailable() ) {
      throw cms::Exception("InvalidConfiguration")
        <<"DataFormat is detected as miniAOD but cannot cast to pat::Photon!";
    }
  }
  
  std::vector<std::vector<float> > float_vars(k_NFloatVars);
  std::vector<std::vector<int> > int_vars(k_NIntVars);
  
  std::unordered_map<std::string,float> float_vars_map;
  std::unordered_map<std::string,int> int_vars_map;
  
  // reco::Photon::superCluster() is virtual so we can exploit polymorphism
  for (unsigned int idxpho = 0; idxpho < src->size(); ++idxpho) {
    const auto& iPho = src->ptrAt(idxpho);

    //    
    // Compute full 5x5 quantities
    //
    const auto& theseed = *(iPho->superCluster()->seed());
    
    if( use_full5x5_ ) {
      calculateValues<noZS::EcalClusterLazyTools>(lazyTools.get(),
                                                  theseed,
                                                  float_vars_map,
                                                  int_vars_map);
    } else {
      calculateValues<EcalClusterLazyTools>(lazyTools.get(),
                                            theseed,
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
  
  lazyTools.reset(nullptr);
}

void PhotonRegressionValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PhotonRegressionValueMapProducer);

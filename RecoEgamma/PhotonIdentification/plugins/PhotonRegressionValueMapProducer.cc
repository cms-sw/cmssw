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
  inline void check_map(const std::unordered_map<std::string,T>& map, unsigned exp_size) {
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

  template<typename T>
  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::Photon> > & handle,
		     const std::vector<T> & values,
		     const std::string    & label) const ;
  
  // The object that will compute 5x5 quantities  
  std::unique_ptr<EcalClusterLazyToolsBase> lazyTools;

  // for AOD case
  edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollection_;
  edm::EDGetToken src_;

  // for miniAOD case
  edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollectionMiniAOD_;
  edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollectionMiniAOD_;
  edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollectionMiniAOD_;
  edm::EDGetToken srcMiniAOD_;
  
  const bool use_full5x5_;
};

PhotonRegressionValueMapProducer::PhotonRegressionValueMapProducer(const edm::ParameterSet& iConfig) :
  use_full5x5_(iConfig.getParameter<bool>("useFull5x5")) {

  //
  // Declare consummables, handle both AOD and miniAOD case
  //
  ebReducedRecHitCollection_        = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("ebReducedRecHitCollection"));
  ebReducedRecHitCollectionMiniAOD_ = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("ebReducedRecHitCollectionMiniAOD"));
  
  eeReducedRecHitCollection_        = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("eeReducedRecHitCollection"));
  eeReducedRecHitCollectionMiniAOD_ = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("eeReducedRecHitCollectionMiniAOD"));
  
  esReducedRecHitCollection_        = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("esReducedRecHitCollection"));
  esReducedRecHitCollectionMiniAOD_ = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("esReducedRecHitCollectionMiniAOD"));
  
  // reco photons are castable into pat photons, so no need to handle reco/pat seprately
  src_        = mayConsume<edm::View<reco::Photon> >(iConfig.getParameter<edm::InputTag>("src"));
  srcMiniAOD_ = mayConsume<edm::View<reco::Photon> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"));

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

  using namespace edm;

  edm::Handle<edm::View<reco::Photon> > src;

  bool isAOD = true; 
  iEvent.getByToken(src_, src);
  if(!src.isValid() ){
    isAOD = false;
    iEvent.getByToken(srcMiniAOD_, src);
  }
  
  if( !src.isValid() ) {
    throw cms::Exception("IllDefinedDataTier")
      << "DataFormat does not contain a photon source!";
  }

  // configure lazy tools
  edm::EDGetTokenT<EcalRecHitCollection> ebrh, eerh, esrh;

  if( isAOD ) {
    ebrh = ebReducedRecHitCollection_;
    eerh = eeReducedRecHitCollection_;
    esrh = esReducedRecHitCollection_;
  } else {
    ebrh = ebReducedRecHitCollectionMiniAOD_;
    eerh = eeReducedRecHitCollectionMiniAOD_;
    esrh = esReducedRecHitCollectionMiniAOD_;
  }
  
  if( use_full5x5_ ) {
    lazyTools = std::make_unique<noZS::EcalClusterLazyTools>( iEvent, iSetup, 
                                                              ebrh, eerh, esrh );
  } else {
    lazyTools = std::make_unique<EcalClusterLazyTools>( iEvent, iSetup, 
                                                        ebrh, eerh, esrh );
  }
  
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
  for (unsigned idxpho = 0; idxpho < src->size(); ++idxpho) {
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
    
    for( unsigned i = 0; i < float_vars.size(); ++i ) {
      float_vars[i].emplace_back(float_vars_map.at(float_var_names[i]));
    }

    
    for( unsigned i = 0; i < int_vars.size(); ++i ) {
      int_vars[i].emplace_back(int_vars_map.at(integer_var_names[i]));
    }
    
  }
  
  for( unsigned i = 0; i < float_vars.size(); ++i ) {
    writeValueMap(iEvent, src, float_vars[i], float_var_names[i]);
  }  
  
  for( unsigned i = 0; i < int_vars.size(); ++i ) {
    writeValueMap(iEvent, src, int_vars[i], integer_var_names[i]);
  }
  
  lazyTools.reset(nullptr);
}

template<typename T>
void PhotonRegressionValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<reco::Photon> > & handle,
					     const std::vector<T> & values,
					     const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  typedef ValueMap<T> TValueMap;
  
  auto valMap = std::make_unique<TValueMap>();
  typename TValueMap::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

void PhotonRegressionValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PhotonRegressionValueMapProducer);

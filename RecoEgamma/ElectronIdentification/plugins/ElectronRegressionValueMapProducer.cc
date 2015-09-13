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

#include "TVector2.h"

#include <memory>
#include <vector>
#include <unordered_map>

namespace {
  enum reg_float_vars { k_sigmaIEtaIPhi = 0,
                        k_eMax,
                        k_e2nd,
                        k_eTop,
                        k_eBottom,
                        k_eLeft,
                        k_eRight,
                        k_clusterMaxDR,
                        k_clusterMaxDRDPhi,
                        k_clusterMaxDRDEta,
                        k_clusterMaxDRRawEnergy,
                        k_clusterRawEnergy0,
                        k_clusterRawEnergy1,
                        k_clusterRawEnergy2,
                        k_clusterDPhiToSeed0,
                        k_clusterDPhiToSeed1,
                        k_clusterDPhiToSeed2,
                        k_clusterDEtaToSeed0,
                        k_clusterDEtaToSeed1,
                        k_clusterDEtaToSeed2,
                        k_cryPhi,
                        k_cryEta,
                        k_NFloatVars             };
  
  enum reg_int_vars { k_iPhi = 0,
                      k_iEta,
                      k_NIntVars     };

  static const std::vector<std::string> float_var_names( { "sigmaIEtaIPhi",
                                                            "eMax",
                                                            "e2nd",
                                                            "eTop",
                                                            "eBottom",
                                                            "eLeft",
                                                            "eRight",
                                                            "clusterMaxDR",
                                                            "clusterMaxDRDPhi",
                                                            "clusterMaxDRDEta",
                                                            "clusterMaxDRRawEnergy",
                                                            "clusterRawEnergy0",
                                                            "clusterRawEnergy1",
                                                            "clusterRawEnergy2",
                                                            "clusterDPhiToSeed0",
                                                            "clusterDPhiToSeed1",
                                                            "clusterDPhiToSeed2",
                                                            "clusterDEtaToSeed0",
                                                            "clusterDEtaToSeed1",
                                                            "clusterDEtaToSeed2",
                                                            "cryPhi",
                                                            "cryEta"                 } );
  
  static const std::vector<std::string> integer_var_names( { "iPhi", "iEta" } );  
  
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
    LazyTools* tools = static_cast<LazyTools*>(tools_tocast);
    
    const auto& the_sc  = iEle->superCluster();
    const auto& theseed = the_sc->seed();
    
    const int numberOfClusters =  the_sc->clusters().size();
    const bool missing_clusters = !the_sc->clusters()[numberOfClusters-1].isAvailable();
    
    std::vector<float> vCov = tools->localCovariances( *theseed );
    
    const float eMax = tools->eMax( *theseed );
    const float e2nd = tools->e2nd( *theseed );
    const float eTop = tools->eTop( *theseed );
    const float eLeft = tools->eLeft( *theseed );
    const float eRight = tools->eRight( *theseed );
    const float eBottom = tools->eBottom( *theseed );
    
    float dummy;
    int iPhi;
    int iEta;
    float cryPhi;
    float cryEta;
    EcalClusterLocal _ecalLocal;
    if (iEle->isEB()) 
      _ecalLocal.localCoordsEB(*theseed, iSetup, cryEta, cryPhi, iEta, iPhi, dummy, dummy);
    else 
      _ecalLocal.localCoordsEE(*theseed, iSetup, cryEta, cryPhi, iEta, iPhi, dummy, dummy);
    
    double see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
    double spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
    double sep;    
    if (see*spp > 0)
      sep = vCov[1] / (see * spp);
    else if (vCov[1] > 0)
      sep = 1.0;
    else
      sep = -1.0;
    
    set_map_val(k_sigmaIEtaIPhi,sep,float_vars);
    set_map_val(k_eMax,eMax,float_vars);
    set_map_val(k_e2nd,e2nd,float_vars);
    set_map_val(k_eTop,eTop,float_vars);
    set_map_val(k_eBottom,eBottom,float_vars);
    set_map_val(k_eLeft,eLeft,float_vars);
    set_map_val(k_eRight,eRight,float_vars);
    set_map_val(k_cryPhi,cryPhi,float_vars);
    set_map_val(k_cryEta,cryEta,float_vars);

    set_map_val(k_iPhi,iPhi,int_vars);
    set_map_val(k_iEta,iEta,int_vars);
    
    std::vector<float> _clusterRawEnergy;
    _clusterRawEnergy.resize(std::max(3, numberOfClusters), 0);
    std::vector<float> _clusterDEtaToSeed;
    _clusterDEtaToSeed.resize(std::max(3, numberOfClusters), 0);
    std::vector<float> _clusterDPhiToSeed;
    _clusterDPhiToSeed.resize(std::max(3, numberOfClusters), 0);
    float _clusterMaxDR     = 999.;
    float _clusterMaxDRDPhi = 999.;
    float _clusterMaxDRDEta = 999.;
    float _clusterMaxDRRawEnergy = 0.;
    
    size_t iclus = 0;
    float maxDR = 0;
    edm::Ptr<reco::CaloCluster> pclus;
    if( !missing_clusters ) {
      // loop over all clusters that aren't the seed  
      auto clusend = the_sc->clustersEnd();
      for( auto clus = the_sc->clustersBegin(); clus != clusend; ++clus ) {
        pclus = *clus;
      
        if(theseed == pclus ) 
          continue;
        _clusterRawEnergy[iclus]  = pclus->energy();
        _clusterDPhiToSeed[iclus] = reco::deltaPhi(pclus->phi(),theseed->phi());
        _clusterDEtaToSeed[iclus] = pclus->eta() - theseed->eta();
        
        // find cluster with max dR
        if(reco::deltaR(*pclus, *theseed) > maxDR) {
          maxDR = reco::deltaR(*pclus, *theseed);
          _clusterMaxDR = maxDR;
          _clusterMaxDRDPhi = _clusterDPhiToSeed[iclus];
          _clusterMaxDRDEta = _clusterDEtaToSeed[iclus];
          _clusterMaxDRRawEnergy = _clusterRawEnergy[iclus];
        }      
        ++iclus;
      }
    }
    
    set_map_val(k_clusterMaxDR,_clusterMaxDR,float_vars);
    set_map_val(k_clusterMaxDRDPhi,_clusterMaxDRDPhi,float_vars);
    set_map_val(k_clusterMaxDRDEta,_clusterMaxDRDEta,float_vars);
    set_map_val(k_clusterMaxDRRawEnergy,_clusterMaxDRRawEnergy,float_vars);
    set_map_val(k_clusterRawEnergy0,_clusterRawEnergy[0],float_vars); 
    set_map_val(k_clusterRawEnergy1,_clusterRawEnergy[1],float_vars); 
    set_map_val(k_clusterRawEnergy2,_clusterRawEnergy[2],float_vars); 
    set_map_val(k_clusterDPhiToSeed0,_clusterDPhiToSeed[0],float_vars);
    set_map_val(k_clusterDPhiToSeed1,_clusterDPhiToSeed[1],float_vars);
    set_map_val(k_clusterDPhiToSeed2,_clusterDPhiToSeed[2],float_vars);
    set_map_val(k_clusterDEtaToSeed0,_clusterDEtaToSeed[0],float_vars);
    set_map_val(k_clusterDEtaToSeed1,_clusterDEtaToSeed[1],float_vars);
    set_map_val(k_clusterDEtaToSeed2,_clusterDEtaToSeed[2],float_vars);
  }
}

class ElectronRegressionValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronRegressionValueMapProducer(const edm::ParameterSet&);
  ~ElectronRegressionValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  template<typename T>
  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
		     const std::vector<T> & values,
		     const std::string    & label) const ;  

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

ElectronRegressionValueMapProducer::ElectronRegressionValueMapProducer(const edm::ParameterSet& iConfig) :
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

  src_        = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("src"));
  srcMiniAOD_ = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"));

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

  using namespace edm;
  
  edm::Handle<edm::View<reco::GsfElectron> > src;

  // Retrieve the collection of electrons from the event.
  // If we fail to retrieve the collection with the standard AOD
  // name, we next look for the one with the stndard miniAOD name.
  bool isAOD = true;
  iEvent.getByToken(src_, src);

  if( !src.isValid() ) {
    isAOD = false;
    iEvent.getByToken(srcMiniAOD_,src);
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
    lazyTools = std::make_unique<noZS::EcalClusterLazyTools>(iEvent, iSetup, 
                                                             ebrh, eerh, esrh );
  } else {
    lazyTools = std::make_unique<EcalClusterLazyTools>(iEvent, iSetup, 
                                                       ebrh, eerh, esrh );
  }

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
  
  lazyTools.reset();
}

template<typename T>
void ElectronRegressionValueMapProducer::writeValueMap(edm::Event &iEvent,
                                                       const edm::Handle<edm::View<reco::GsfElectron> > & handle,
                                                       const std::vector<T> & values,
                                                       const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  typedef ValueMap<T> TValueMap;

  auto_ptr<TValueMap> valMap(new TValueMap());
  typename TValueMap::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(valMap, label);
}

void ElectronRegressionValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronRegressionValueMapProducer);

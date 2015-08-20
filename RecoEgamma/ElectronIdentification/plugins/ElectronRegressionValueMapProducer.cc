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

namespace {
  constexpr char sigmaIEtaIPhi_[] = "sigmaIEtaIPhi";
  constexpr char eMax_[] = "eMax";
  constexpr char e2nd_[] = "e2nd";
  constexpr char eTop_[] = "eTop";
  constexpr char eBottom_[] = "eBottom";
  constexpr char eLeft_[] = "eLeft";
  constexpr char eRight_[] = "eRight";
  constexpr char clusterMaxDR_[] = "clusterMaxDR";
  constexpr char clusterMaxDRDPhi_[] = "clusterMaxDRDPhi";
  constexpr char clusterMaxDRDEta_[] = "clusterMaxDRDEta";
  constexpr char clusterMaxDRRawEnergy_[] = "clusterMaxDRRawEnergy";
  constexpr char clusterRawEnergy0_[] = "clusterRawEnergy0";
  constexpr char clusterRawEnergy1_[] = "clusterRawEnergy1";
  constexpr char clusterRawEnergy2_[] = "clusterRawEnergy2";
  constexpr char clusterDPhiToSeed0_[] = "clusterDPhiToSeed0";
  constexpr char clusterDPhiToSeed1_[] = "clusterDPhiToSeed1";
  constexpr char clusterDPhiToSeed2_[] = "clusterDPhiToSeed2";
  constexpr char clusterDEtaToSeed0_[] = "clusterDEtaToSeed0";
  constexpr char clusterDEtaToSeed1_[] = "clusterDEtaToSeed1";
  constexpr char clusterDEtaToSeed2_[] = "clusterDEtaToSeed2";
  constexpr char eleIPhi_[]    = "iPhi";
  constexpr char eleIEta_[]    = "iEta";
  constexpr char eleCryPhi_[]  = "cryPhi";
  constexpr char eleCryEta_[]  = "cryEta";
}

class ElectronRegressionValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronRegressionValueMapProducer(const edm::ParameterSet&);
  ~ElectronRegressionValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
		     const std::vector<float> & values,
		     const std::string    & label) const ;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
		     const std::vector<int> & values,
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

  produces<edm::ValueMap<float> >(sigmaIEtaIPhi_); 
  produces<edm::ValueMap<float> >(eMax_);
  produces<edm::ValueMap<float> >(e2nd_);
  produces<edm::ValueMap<float> >(eTop_);
  produces<edm::ValueMap<float> >(eBottom_);
  produces<edm::ValueMap<float> >(eLeft_);
  produces<edm::ValueMap<float> >(eRight_);
  produces<edm::ValueMap<float> >(clusterMaxDR_);
  produces<edm::ValueMap<float> >(clusterMaxDRDPhi_);
  produces<edm::ValueMap<float> >(clusterMaxDRDEta_);
  produces<edm::ValueMap<float> >(clusterMaxDRRawEnergy_);
  produces<edm::ValueMap<float> >(clusterRawEnergy0_); 
  produces<edm::ValueMap<float> >(clusterRawEnergy1_); 
  produces<edm::ValueMap<float> >(clusterRawEnergy2_); 
  produces<edm::ValueMap<float> >(clusterDPhiToSeed0_);
  produces<edm::ValueMap<float> >(clusterDPhiToSeed1_);
  produces<edm::ValueMap<float> >(clusterDPhiToSeed2_);
  produces<edm::ValueMap<float> >(clusterDEtaToSeed0_);
  produces<edm::ValueMap<float> >(clusterDEtaToSeed1_);
  produces<edm::ValueMap<float> >(clusterDEtaToSeed2_);
  produces<edm::ValueMap<int> >(eleIPhi_);
  produces<edm::ValueMap<int> >(eleIEta_);
  produces<edm::ValueMap<float> >(eleCryPhi_);
  produces<edm::ValueMap<float> >(eleCryEta_);
}

ElectronRegressionValueMapProducer::~ElectronRegressionValueMapProducer() {
}

template<typename LazyTools>
inline void calculateValues(EcalClusterLazyToolsBase* tools_tocast,
                            const edm::Ptr<reco::GsfElectron>& iEle,
                            const edm::EventSetup& iSetup,
                            std::vector<float>& vsigmaIEtaIPhi,
                            std::vector<float>& veMax,
                            std::vector<float>& ve2nd,
                            std::vector<float>& veTop,
                            std::vector<float>& veBottom,
                            std::vector<float>& veLeft,
                            std::vector<float>& veRight,
                            std::vector<float>& vclusterMaxDR,
                            std::vector<float>& vclusterMaxDRDPhi,
                            std::vector<float>& vclusterMaxDRDEta,
                            std::vector<float>& vclusterMaxDRRawEnergy,
                            std::vector<float>& vclusterRawEnergy0, 
                            std::vector<float>& vclusterRawEnergy1, 
                            std::vector<float>& vclusterRawEnergy2, 
                            std::vector<float>& vclusterDPhiToSeed0,
                            std::vector<float>& vclusterDPhiToSeed1,
                            std::vector<float>& vclusterDPhiToSeed2,
                            std::vector<float>& vclusterDEtaToSeed0,
                            std::vector<float>& vclusterDEtaToSeed1,
                            std::vector<float>& vclusterDEtaToSeed2,
                            std::vector<int>& veleIPhi,
                            std::vector<int>& veleIEta,
                            std::vector<float>& veleCryPhi,
                            std::vector<float>& veleCryEta) {
  LazyTools* tools = static_cast<LazyTools*>(tools_tocast);
  
  const auto& the_sc  = iEle->superCluster();
  const auto& theseed = the_sc->seed();
  
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
  
  vsigmaIEtaIPhi.push_back(sep);
  veMax.push_back(eMax);
  ve2nd.push_back(e2nd);
  veTop.push_back(eTop);
  veBottom.push_back(eBottom);
  veLeft.push_back(eLeft);
  veRight.push_back(eRight);
  veleIPhi.push_back(iPhi);
  veleIEta.push_back(iEta);
  veleCryPhi.push_back(cryPhi);
  veleCryEta.push_back(cryEta);
  
    // loop over all clusters that aren't the seed
  auto clusend = the_sc->clustersEnd();
  int numberOfClusters =  the_sc->clusters().size();
  
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
  for( auto clus = the_sc->clustersBegin(); clus != clusend; ++clus ) {
    pclus = *clus;
    
    if(theseed == pclus ) 
      continue;
    _clusterRawEnergy.push_back(pclus->energy());
    _clusterDPhiToSeed.push_back(reco::deltaPhi(pclus->phi(),theseed->phi()));
    _clusterDEtaToSeed.push_back(pclus->eta() - theseed->eta());
    
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
  
  vclusterMaxDR.push_back(_clusterMaxDR);
  vclusterMaxDRDPhi.push_back(_clusterMaxDRDPhi);
  vclusterMaxDRDEta.push_back(_clusterMaxDRDEta);
  vclusterMaxDRRawEnergy.push_back(_clusterMaxDRRawEnergy);
  vclusterRawEnergy0.push_back(_clusterRawEnergy[0]); 
  vclusterRawEnergy1.push_back(_clusterRawEnergy[1]); 
  vclusterRawEnergy2.push_back(_clusterRawEnergy[2]); 
  vclusterDPhiToSeed0.push_back(_clusterDPhiToSeed[0]);
  vclusterDPhiToSeed1.push_back(_clusterDPhiToSeed[1]);
  vclusterDPhiToSeed2.push_back(_clusterDPhiToSeed[2]);
  vclusterDEtaToSeed0.push_back(_clusterDEtaToSeed[0]);
  vclusterDEtaToSeed1.push_back(_clusterDEtaToSeed[1]);
  vclusterDEtaToSeed2.push_back(_clusterDEtaToSeed[2]);
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
    lazyTools.reset( new noZS::EcalClusterLazyTools(iEvent, iSetup, 
                                                    ebrh, eerh, esrh ) );
  } else {
    lazyTools.reset( new EcalClusterLazyTools(iEvent, iSetup, 
                                              ebrh, eerh, esrh ) );
  }

  std::vector<float> sigmaIEtaIPhi;
  std::vector<float> eMax;
  std::vector<float> e2nd;
  std::vector<float> eTop;
  std::vector<float> eBottom;
  std::vector<float> eLeft;
  std::vector<float> eRight;
  std::vector<float> clusterMaxDR;
  std::vector<float> clusterMaxDRDPhi;
  std::vector<float> clusterMaxDRDEta;
  std::vector<float> clusterMaxDRRawEnergy;
  std::vector<float> clusterRawEnergy0; 
  std::vector<float> clusterRawEnergy1; 
  std::vector<float> clusterRawEnergy2; 
  std::vector<float> clusterDPhiToSeed0;
  std::vector<float> clusterDPhiToSeed1;
  std::vector<float> clusterDPhiToSeed2;
  std::vector<float> clusterDEtaToSeed0;
  std::vector<float> clusterDEtaToSeed1;
  std::vector<float> clusterDEtaToSeed2;
  std::vector<int> eleIPhi;
  std::vector<int> eleIEta;
  std::vector<float> eleCryPhi;
  std::vector<float> eleCryEta;

  // reco::GsfElectron::superCluster() is virtual so we can exploit polymorphism
  for (size_t i = 0; i < src->size(); ++i){
    auto iEle = src->ptrAt(i);

    if( use_full5x5_ ) {
      calculateValues<noZS::EcalClusterLazyTools>(lazyTools.get(),
                                                  iEle,
                                                  iSetup,
                                                  sigmaIEtaIPhi,
                                                  eMax,
                                                  e2nd,
                                                  eTop,
                                                  eBottom,
                                                  eLeft,
                                                  eRight,
                                                  clusterMaxDR,
                                                  clusterMaxDRDPhi,
                                                  clusterMaxDRDEta,
                                                  clusterMaxDRRawEnergy,
                                                  clusterRawEnergy0, 
                                                  clusterRawEnergy1, 
                                                  clusterRawEnergy2, 
                                                  clusterDPhiToSeed0,
                                                  clusterDPhiToSeed1,
                                                  clusterDPhiToSeed2,
                                                  clusterDEtaToSeed0,
                                                  clusterDEtaToSeed1,
                                                  clusterDEtaToSeed2,
                                                  eleIPhi,
                                                  eleIEta,
                                                  eleCryPhi,
                                                  eleCryEta);
    } else {
      calculateValues<EcalClusterLazyTools>(lazyTools.get(),
                                            iEle,
                                            iSetup,
                                            sigmaIEtaIPhi,
                                            eMax,
                                            e2nd,
                                            eTop,
                                            eBottom,
                                            eLeft,
                                            eRight,
                                            clusterMaxDR,
                                            clusterMaxDRDPhi,
                                            clusterMaxDRDEta,
                                            clusterMaxDRRawEnergy,
                                            clusterRawEnergy0, 
                                            clusterRawEnergy1, 
                                            clusterRawEnergy2, 
                                            clusterDPhiToSeed0,
                                            clusterDPhiToSeed1,
                                            clusterDPhiToSeed2,
                                            clusterDEtaToSeed0,
                                            clusterDEtaToSeed1,
                                            clusterDEtaToSeed2,
                                            eleIPhi,
                                            eleIEta,
                                            eleCryPhi,
                                            eleCryEta);
    }
  }
  
  writeValueMap(iEvent, src, sigmaIEtaIPhi, sigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, eMax      ,eMax_);
  writeValueMap(iEvent, src, e2nd	 ,e2nd_);
  writeValueMap(iEvent, src, eTop	 ,eTop_);
  writeValueMap(iEvent, src, eBottom	 ,eBottom_);
  writeValueMap(iEvent, src, eLeft     ,eLeft_);
  writeValueMap(iEvent, src, eRight	 ,eRight_);
  writeValueMap(iEvent, src, clusterMaxDR,	   clusterMaxDR_);	  
  writeValueMap(iEvent, src, clusterMaxDRDPhi,	   clusterMaxDRDPhi_);	  
  writeValueMap(iEvent, src, clusterMaxDRDEta,	   clusterMaxDRDEta_);	  
  writeValueMap(iEvent, src, clusterMaxDRRawEnergy,clusterMaxDRRawEnergy_);
  writeValueMap(iEvent, src, clusterRawEnergy0,    clusterRawEnergy0_);   
  writeValueMap(iEvent, src, clusterRawEnergy1,    clusterRawEnergy1_);   
  writeValueMap(iEvent, src, clusterRawEnergy2,    clusterRawEnergy2_);   
  writeValueMap(iEvent, src, clusterDPhiToSeed0,   clusterDPhiToSeed0_);  
  writeValueMap(iEvent, src, clusterDPhiToSeed1,   clusterDPhiToSeed1_);  
  writeValueMap(iEvent, src, clusterDPhiToSeed2,   clusterDPhiToSeed2_);  
  writeValueMap(iEvent, src, clusterDEtaToSeed0,   clusterDEtaToSeed0_);  
  writeValueMap(iEvent, src, clusterDEtaToSeed1,   clusterDEtaToSeed1_);  
  writeValueMap(iEvent, src, clusterDEtaToSeed2,   clusterDEtaToSeed2_);  
  writeValueMap(iEvent, src, eleIPhi		 ,eleIPhi_);
  writeValueMap(iEvent, src, eleIEta		 ,eleIEta_);
  writeValueMap(iEvent, src, eleCryPhi		 ,eleCryPhi_);
  writeValueMap(iEvent, src, eleCryEta           ,eleCryEta_);
  lazyTools.reset();
}

void ElectronRegressionValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
					     const std::vector<float> & values,
					     const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(valMap, label);
}

void ElectronRegressionValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
					     const std::vector<int> & values,
					     const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  auto_ptr<ValueMap<int> > valMap(new ValueMap<int>());
  edm::ValueMap<int>::Filler filler(*valMap);
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

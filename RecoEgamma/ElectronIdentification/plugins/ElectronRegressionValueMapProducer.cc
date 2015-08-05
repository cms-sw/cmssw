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

  noZS::EcalClusterLazyTools *lazyToolnoZS;

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

  constexpr static char eleFull5x5SigmaIEtaIPhi_[] = "eleFull5x5SigmaIEtaIPhi";
  constexpr static char eleFull5x5eMax_[] = "eMax";
  constexpr static char eleFull5x5e2nd_[] = "e2nd";
  constexpr static char eleFull5x5eTop_[] = "eTop";
  constexpr static char eleFull5x5eBottom_[] = "eBottom";
  constexpr static char eleFull5x5eLeft_[] = "eLeft";
  constexpr static char eleFull5x5eRight_[] = "eRight";
  constexpr static char  clusterMaxDR_[] = "clusterMaxDR";
  constexpr static char  clusterMaxDRDPhi_[] = "clusterMaxDRDPhi";
  constexpr static char  clusterMaxDRDEta_[] = "clusterMaxDRDEta";
  constexpr static char  clusterMaxDRRawEnergy_[] = "clusterMaxDRRawEnergy";
  constexpr static char  clusterRawEnergy0_[] = "clusterRawEnergy0";
  constexpr static char  clusterRawEnergy1_[] = "clusterRawEnergy1";
  constexpr static char  clusterRawEnergy2_[] = "clusterRawEnergy2";
  constexpr static char  clusterDPhiToSeed0_[] = "clusterDPhiToSeed0";
  constexpr static char  clusterDPhiToSeed1_[] = "clusterDPhiToSeed1";
  constexpr static char  clusterDPhiToSeed2_[] = "clusterDPhiToSeed2";
  constexpr static char  clusterDEtaToSeed0_[] = "clusterDEtaToSeed0";
  constexpr static char  clusterDEtaToSeed1_[] = "clusterDEtaToSeed1";
  constexpr static char  clusterDEtaToSeed2_[] = "clusterDEtaToSeed2";
  constexpr static char eleIPhi_[]    = "iPhi";
  constexpr static char eleIEta_[]    = "iEta";
  constexpr static char eleCryPhi_[]  = "cryPhi";
  constexpr static char eleCryEta_[]  = "cryEta";
};

constexpr char ElectronRegressionValueMapProducer::eleFull5x5SigmaIEtaIPhi_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5eMax_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5e2nd_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5eTop_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5eBottom_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5eLeft_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5eRight_[];
constexpr char ElectronRegressionValueMapProducer::clusterMaxDR_[];
constexpr char ElectronRegressionValueMapProducer::clusterMaxDRDPhi_[];
constexpr char ElectronRegressionValueMapProducer::clusterMaxDRDEta_[];
constexpr char ElectronRegressionValueMapProducer::clusterMaxDRRawEnergy_[];
constexpr char ElectronRegressionValueMapProducer::clusterRawEnergy0_[]; 
constexpr char ElectronRegressionValueMapProducer::clusterRawEnergy1_[]; 
constexpr char ElectronRegressionValueMapProducer::clusterRawEnergy2_[]; 
constexpr char ElectronRegressionValueMapProducer::clusterDPhiToSeed0_[];
constexpr char ElectronRegressionValueMapProducer::clusterDPhiToSeed1_[];
constexpr char ElectronRegressionValueMapProducer::clusterDPhiToSeed2_[];
constexpr char ElectronRegressionValueMapProducer::clusterDEtaToSeed0_[];
constexpr char ElectronRegressionValueMapProducer::clusterDEtaToSeed1_[];
constexpr char ElectronRegressionValueMapProducer::clusterDEtaToSeed2_[];
constexpr char ElectronRegressionValueMapProducer::eleIPhi_[];
constexpr char ElectronRegressionValueMapProducer::eleIEta_[];
constexpr char ElectronRegressionValueMapProducer::eleCryPhi_[];
constexpr char ElectronRegressionValueMapProducer::eleCryEta_[];

ElectronRegressionValueMapProducer::ElectronRegressionValueMapProducer(const edm::ParameterSet& iConfig) {

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

  produces<edm::ValueMap<float> >(eleFull5x5SigmaIEtaIPhi_); 
  produces<edm::ValueMap<float> >(eleFull5x5eMax_);
  produces<edm::ValueMap<float> >(eleFull5x5e2nd_);
  produces<edm::ValueMap<float> >(eleFull5x5eTop_);
  produces<edm::ValueMap<float> >(eleFull5x5eBottom_);
  produces<edm::ValueMap<float> >(eleFull5x5eLeft_);
  produces<edm::ValueMap<float> >(eleFull5x5eRight_);
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

void ElectronRegressionValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  
  edm::Handle<edm::View<reco::GsfElectron> > src;

  // Retrieve the collection of electrons from the event.
  // If we fail to retrieve the collection with the standard AOD
  // name, we next look for the one with the stndard miniAOD name.
  bool isAOD = true;
  iEvent.getByToken(src_, src);

  if( !src.isValid() ){
    isAOD = false;
    iEvent.getByToken(srcMiniAOD_,src);
  }

  if( isAOD )
    lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, 
						  ebReducedRecHitCollection_, 
						  eeReducedRecHitCollection_,
						  esReducedRecHitCollection_ );
  else
    lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, 
						  ebReducedRecHitCollectionMiniAOD_, 
						  eeReducedRecHitCollectionMiniAOD_,
						  esReducedRecHitCollectionMiniAOD_ );
 
  std::vector<float> eleFull5x5SigmaIEtaIPhi;
  std::vector<float> eleFull5x5eMax;
  std::vector<float> eleFull5x5e2nd;
  std::vector<float> eleFull5x5eTop;
  std::vector<float> eleFull5x5eBottom;
  std::vector<float> eleFull5x5eLeft;
  std::vector<float> eleFull5x5eRight;
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
    const auto theseed = iEle->superCluster()->seed();

    std::vector<float> vCov = lazyToolnoZS->localCovariances( *theseed );
    float sep = vCov[1];
        
    const float eMax = lazyToolnoZS->eMax( *theseed );
    const float e2nd = lazyToolnoZS->e2nd( *theseed );
    const float eTop = lazyToolnoZS->eTop( *theseed );
    const float eLeft = lazyToolnoZS->eLeft( *theseed );
    const float eRight = lazyToolnoZS->eRight( *theseed );
    const float eBottom = lazyToolnoZS->eBottom( *theseed );
    
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
    if (see*spp > 0)
        sep = vCov[1] / (see * spp);
    else if (vCov[1] > 0)
        sep = 1.0;
    else
        sep = -1.0;
 
    eleFull5x5SigmaIEtaIPhi.push_back(sep);
    eleFull5x5eMax.push_back(eMax);
    eleFull5x5e2nd.push_back(e2nd);
    eleFull5x5eTop.push_back(eTop);
    eleFull5x5eBottom.push_back(eBottom);
    eleFull5x5eLeft.push_back(eLeft);
    eleFull5x5eRight.push_back(eRight);
    eleIPhi.push_back(iPhi);
    eleIEta.push_back(iEta);
    eleCryPhi.push_back(cryPhi);
    eleCryEta.push_back(cryEta);

    // loop over all clusters that aren't the seed
    auto clusend = iEle->superCluster()->clustersEnd();
    int numberOfClusters =  iEle->superCluster()->clusters().size();

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
    for( auto clus = iEle->superCluster()->clustersBegin(); clus != clusend; ++clus ) {
      pclus = *clus;
            
      if(theseed == pclus ) 
	continue;
      _clusterRawEnergy.push_back(pclus->energy());
      _clusterDPhiToSeed.push_back(TVector2::Phi_mpi_pi(pclus->phi() - theseed->phi()));
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
  
    clusterMaxDR.push_back(_clusterMaxDR);
    clusterMaxDRDPhi.push_back(_clusterMaxDRDPhi);
    clusterMaxDRDEta.push_back(_clusterMaxDRDEta);
    clusterMaxDRRawEnergy.push_back(_clusterMaxDRRawEnergy);
    clusterRawEnergy0.push_back(_clusterRawEnergy[0]); 
    clusterRawEnergy1.push_back(_clusterRawEnergy[1]); 
    clusterRawEnergy2.push_back(_clusterRawEnergy[2]); 
    clusterDPhiToSeed0.push_back(_clusterDPhiToSeed[0]);
    clusterDPhiToSeed1.push_back(_clusterDPhiToSeed[1]);
    clusterDPhiToSeed2.push_back(_clusterDPhiToSeed[2]);
    clusterDEtaToSeed0.push_back(_clusterDEtaToSeed[0]);
    clusterDEtaToSeed1.push_back(_clusterDEtaToSeed[1]);
    clusterDEtaToSeed2.push_back(_clusterDEtaToSeed[2]);
  }
  
  writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIPhi, eleFull5x5SigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, eleFull5x5eMax      ,eleFull5x5eMax_);
  writeValueMap(iEvent, src, eleFull5x5e2nd	 ,eleFull5x5e2nd_);
  writeValueMap(iEvent, src, eleFull5x5eTop	 ,eleFull5x5eTop_);
  writeValueMap(iEvent, src, eleFull5x5eBottom	 ,eleFull5x5eBottom_);
  writeValueMap(iEvent, src, eleFull5x5eLeft     ,eleFull5x5eLeft_);
  writeValueMap(iEvent, src, eleFull5x5eRight	 ,eleFull5x5eRight_);
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
  delete lazyToolnoZS;
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

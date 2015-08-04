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
  constexpr static char eleFull5x5e2x5Top_[]    = "e2x5Top";
  constexpr static char eleFull5x5e2x5Bottom_[] = "e2x5Bottom";
  constexpr static char eleFull5x5e2x5Left_[]   = "e2x5Left";
  constexpr static char eleFull5x5e2x5Right_[]  = "e2x5Right";
  constexpr static char eleFull5x5e3x3_[]  = "e3x3";
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
constexpr char ElectronRegressionValueMapProducer::eleFull5x5e2x5Top_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5e2x5Bottom_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5e2x5Left_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5e2x5Right_[];
constexpr char ElectronRegressionValueMapProducer::eleFull5x5e3x3_[];
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
  produces<edm::ValueMap<float> >(eleFull5x5e2x5Top_);
  produces<edm::ValueMap<float> >(eleFull5x5e2x5Bottom_);
  produces<edm::ValueMap<float> >(eleFull5x5e2x5Left_);
  produces<edm::ValueMap<float> >(eleFull5x5e2x5Right_);
  produces<edm::ValueMap<float> >(eleFull5x5e3x3_);
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
  std::vector<float> eleFull5x5e2x5Top;
  std::vector<float> eleFull5x5e2x5Bottom;
  std::vector<float> eleFull5x5e2x5Left;
  std::vector<float> eleFull5x5e2x5Right;
  std::vector<float> eleFull5x5e3x3;
  std::vector<int> eleIPhi;
  std::vector<int> eleIEta;
  std::vector<float> eleCryPhi;
  std::vector<float> eleCryEta;

  // reco::GsfElectron::superCluster() is virtual so we can exploit polymorphism
  for (size_t i = 0; i < src->size(); ++i){
    auto iEle = src->ptrAt(i);
    const auto& theseed = *(iEle->superCluster()->seed());

    std::vector<float> vCov = lazyToolnoZS->localCovariances( theseed );
    const float sep = vCov[1];
        
    const float eMax = lazyToolnoZS->eMax( theseed );
    const float e2nd = lazyToolnoZS->e2nd( theseed );
    const float eTop = lazyToolnoZS->eTop( theseed );
    const float eLeft = lazyToolnoZS->eLeft( theseed );
    const float eRight = lazyToolnoZS->eRight( theseed );
    const float eBottom = lazyToolnoZS->eBottom( theseed );
    const float e3x3 = lazyToolnoZS->e3x3( theseed );
    const float e2x5Top = lazyToolnoZS->e2x5Top( theseed );
    const float e2x5Left = lazyToolnoZS->e2x5Left( theseed );
    const float e2x5Right = lazyToolnoZS->e2x5Right( theseed );
    const float e2x5Bottom = lazyToolnoZS->e2x5Bottom( theseed );
    
    float dummy;
    int iPhi;
    int iEta;
    float cryPhi;
    float cryEta;
    EcalClusterLocal _ecalLocal;
    if (iEle->isEB()) 
      _ecalLocal.localCoordsEB(theseed, iSetup, cryEta, cryPhi, iEta, iPhi, dummy, dummy);
    else 
      _ecalLocal.localCoordsEE(theseed, iSetup, cryEta, cryPhi, iEta, iPhi, dummy, dummy);
    
    eleFull5x5SigmaIEtaIPhi.push_back(sep);
    eleFull5x5eMax.push_back(eMax);
    eleFull5x5e2nd.push_back(e2nd);
    eleFull5x5eTop.push_back(eTop);
    eleFull5x5eBottom.push_back(eBottom);
    eleFull5x5eLeft.push_back(eLeft);
    eleFull5x5eRight.push_back(eRight);
    eleFull5x5e2x5Top.push_back(e2x5Top);
    eleFull5x5e2x5Bottom .push_back(e2x5Bottom);
    eleFull5x5e2x5Left.push_back(e2x5Left);
    eleFull5x5e2x5Right.push_back(e2x5Right);
    eleFull5x5e3x3.push_back(e3x3);
    eleIPhi.push_back(iPhi);
    eleIEta.push_back(iEta);
    eleCryPhi.push_back(cryPhi);
    eleCryEta.push_back(cryEta);
  }
  
  writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIPhi, eleFull5x5SigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, eleFull5x5eMax      ,eleFull5x5eMax_);
  writeValueMap(iEvent, src, eleFull5x5e2nd	 ,eleFull5x5e2nd_);
  writeValueMap(iEvent, src, eleFull5x5eTop	 ,eleFull5x5eTop_);
  writeValueMap(iEvent, src, eleFull5x5eBottom	 ,eleFull5x5eBottom_);
  writeValueMap(iEvent, src, eleFull5x5eLeft     ,eleFull5x5eLeft_);
  writeValueMap(iEvent, src, eleFull5x5eRight	 ,eleFull5x5eRight_);
  writeValueMap(iEvent, src, eleFull5x5e2x5Top	 ,eleFull5x5e2x5Top_);
  writeValueMap(iEvent, src, eleFull5x5e2x5Bottom,eleFull5x5e2x5Bottom_);
  writeValueMap(iEvent, src, eleFull5x5e2x5Left  ,eleFull5x5e2x5Left_);
  writeValueMap(iEvent, src, eleFull5x5e2x5Right ,eleFull5x5e2x5Right_);
  writeValueMap(iEvent, src, eleFull5x5e3x3	 ,eleFull5x5e3x3_);
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

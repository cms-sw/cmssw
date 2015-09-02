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

namespace {
  // Cluster shapes
  constexpr char sigmaIPhiIPhi_[] = "sigmaIPhiIPhi";
  constexpr char sigmaIEtaIPhi_[] = "sigmaIEtaIPhi";
  constexpr char e2x5Max_[] = "e2x5Max";
  constexpr char e2x5Left_[] = "e2x5Left";
  constexpr char e2x5Right_[] = "e2x5Right";
  constexpr char e2x5Top_[] = "e2x5Top";
  constexpr char e2x5Bottom_[] = "e2x5Bottom";
}

class PhotonRegressionValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit PhotonRegressionValueMapProducer(const edm::ParameterSet&);
  ~PhotonRegressionValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::Photon> > & handle,
		     const std::vector<float> & values,
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
  produces<edm::ValueMap<float> >(sigmaIPhiIPhi_);  
  produces<edm::ValueMap<float> >(sigmaIEtaIPhi_);  
  produces<edm::ValueMap<float> >(e2x5Max_);  
  produces<edm::ValueMap<float> >(e2x5Left_);  
  produces<edm::ValueMap<float> >(e2x5Right_);  
  produces<edm::ValueMap<float> >(e2x5Top_);  
  produces<edm::ValueMap<float> >(e2x5Bottom_);  
}

PhotonRegressionValueMapProducer::~PhotonRegressionValueMapProducer() 
{}

template<typename LazyTools,typename SeedType>
inline void calculateValues(EcalClusterLazyToolsBase* tools_tocast,
                            const SeedType& the_seed,
                            std::vector<float>& sigmaIPhiIPhi,
                            std::vector<float>& sigmaIEtaIPhi,
                            std::vector<float>& e2x5Max,
                            std::vector<float>& e2x5Left,
                            std::vector<float>& e2x5Right,
                            std::vector<float>& e2x5Top,
                            std::vector<float>& e2x5Bottom) {
  LazyTools* tools = static_cast<LazyTools*>(tools_tocast);
  
  float spp = -999;
  std::vector<float> vCov = tools->localCovariances( the_seed );
  spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
  float sep = vCov[1];
  sigmaIPhiIPhi.push_back(spp);
  sigmaIEtaIPhi.push_back(sep);
  e2x5Max   .push_back(tools->e2x5Max(the_seed) );
  e2x5Left  .push_back(tools->e2x5Left(the_seed) );
  e2x5Right .push_back(tools->e2x5Right(the_seed) );
  e2x5Top   .push_back(tools->e2x5Top(the_seed) );
  e2x5Bottom.push_back(tools->e2x5Bottom(the_seed) );  
}

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
    lazyTools.reset( new noZS::EcalClusterLazyTools(iEvent, iSetup, 
                                                    ebrh, eerh, esrh ) );
  } else {
    lazyTools.reset( new EcalClusterLazyTools(iEvent, iSetup, 
                                              ebrh, eerh, esrh ) );
  }
  
  if( !isAOD && src->size() ) {
    edm::Ptr<pat::Photon> test(src->ptrAt(0));
    if( test.isNull() || !test.isAvailable() ) {
      throw cms::Exception("InvalidConfiguration")
	<<"DataFormat is detected as miniAOD but cannot cast to pat::Photon!";
    }
  }
  
  // Cluster shapes
  std::vector<float> sigmaIPhiIPhi;
  std::vector<float> sigmaIEtaIPhi;
  std::vector<float> e2x5Max;
  std::vector<float> e2x5Left;
  std::vector<float> e2x5Right;
  std::vector<float> e2x5Top;
  std::vector<float> e2x5Bottom;
  
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
                                                  sigmaIPhiIPhi,
                                                  sigmaIEtaIPhi,
                                                  e2x5Max,
                                                  e2x5Left,
                                                  e2x5Right,
                                                  e2x5Top,
                                                  e2x5Bottom);
    } else {
      calculateValues<EcalClusterLazyTools>(lazyTools.get(),
                                            theseed,
                                            sigmaIPhiIPhi,
                                            sigmaIEtaIPhi,
                                            e2x5Max,
                                            e2x5Left,
                                            e2x5Right,
                                            e2x5Top,
                                            e2x5Bottom);
    }   
  }
  
  // Cluster shapes
  writeValueMap(iEvent, src, sigmaIPhiIPhi, sigmaIPhiIPhi_);  
  writeValueMap(iEvent, src, sigmaIEtaIPhi, sigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, e2x5Max, e2x5Max_);  
  writeValueMap(iEvent, src, e2x5Left, e2x5Left_);  
  writeValueMap(iEvent, src, e2x5Right, e2x5Right_);  
  writeValueMap(iEvent, src, e2x5Top, e2x5Top_);  
  writeValueMap(iEvent, src, e2x5Bottom, e2x5Bottom_);  

  lazyTools.reset(nullptr);
}

void PhotonRegressionValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<reco::Photon> > & handle,
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

void PhotonRegressionValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PhotonRegressionValueMapProducer);

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
  std::unique_ptr<noZS::EcalClusterLazyTools> lazyToolnoZS;

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

  // Cluster shapes
  constexpr static char phoFull5x5SigmaIPhiIPhi_[] = "phoFull5x5SigmaIPhiIPhi";
  constexpr static char phoFull5x5SigmaIEtaIPhi_[] = "phoFull5x5SigmaIEtaIPhi";
  constexpr static char phoFull5x5E2x5Max_[] = "phoFull5x5E2x5Max";
  constexpr static char phoFull5x5E2x5Left_[] = "phoFull5x5E2x5Left";
  constexpr static char phoFull5x5E2x5Right_[] = "phoFull5x5E2x5Right";
  constexpr static char phoFull5x5E2x5Top_[] = "phoFull5x5E2x5Top";
  constexpr static char phoFull5x5E2x5Bottom_[] = "phoFull5x5E2x5Bottom";
};

// Cluster shapes
constexpr char PhotonRegressionValueMapProducer::phoFull5x5SigmaIPhiIPhi_[];
constexpr char PhotonRegressionValueMapProducer::phoFull5x5SigmaIEtaIPhi_[];
constexpr char PhotonRegressionValueMapProducer::phoFull5x5E2x5Max_[];
constexpr char PhotonRegressionValueMapProducer::phoFull5x5E2x5Left_[];
constexpr char PhotonRegressionValueMapProducer::phoFull5x5E2x5Right_[];
constexpr char PhotonRegressionValueMapProducer::phoFull5x5E2x5Top_[];
constexpr char PhotonRegressionValueMapProducer::phoFull5x5E2x5Bottom_[];

PhotonRegressionValueMapProducer::PhotonRegressionValueMapProducer(const edm::ParameterSet& iConfig) {

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
  produces<edm::ValueMap<float> >(phoFull5x5SigmaIPhiIPhi_);  
  produces<edm::ValueMap<float> >(phoFull5x5SigmaIEtaIPhi_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x5Max_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x5Left_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x5Right_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x5Top_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x5Bottom_);  
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

  // Configure Lazy Tools
  if( isAOD )
    lazyToolnoZS = std::unique_ptr<noZS::EcalClusterLazyTools>(new noZS::EcalClusterLazyTools(iEvent, iSetup, 
											      ebReducedRecHitCollection_,
											      eeReducedRecHitCollection_,
											      esReducedRecHitCollection_));
  else
    lazyToolnoZS = std::unique_ptr<noZS::EcalClusterLazyTools>(new noZS::EcalClusterLazyTools(iEvent, iSetup, 
											      ebReducedRecHitCollectionMiniAOD_,
											      eeReducedRecHitCollectionMiniAOD_,
											      esReducedRecHitCollectionMiniAOD_)); 

  if( !isAOD && src->size() ) {
    edm::Ptr<pat::Photon> test(src->ptrAt(0));
    if( test.isNull() || !test.isAvailable() ) {
      throw cms::Exception("InvalidConfiguration")
	<<"DataFormat is detected as miniAOD but cannot cast to pat::Photon!";
    }
  }
  
  // Cluster shapes
  std::vector<float> phoFull5x5SigmaIPhiIPhi;
  std::vector<float> phoFull5x5SigmaIEtaIPhi;
  std::vector<float> phoFull5x5E2x5Max;
  std::vector<float> phoFull5x5E2x5Left;
  std::vector<float> phoFull5x5E2x5Right;
  std::vector<float> phoFull5x5E2x5Top;
  std::vector<float> phoFull5x5E2x5Bottom;
  
  // reco::Photon::superCluster() is virtual so we can exploit polymorphism
  for (unsigned idxpho = 0; idxpho < src->size(); ++idxpho) {
    const auto& iPho = src->ptrAt(idxpho);

    //    
    // Compute full 5x5 quantities
    //
    const auto& theseed = *(iPho->superCluster()->seed());
    
    float spp = -999;
    std::vector<float> vCov = lazyToolnoZS->localCovariances( theseed );
    spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
    float sep = vCov[1];
    phoFull5x5SigmaIPhiIPhi.push_back(spp);
    phoFull5x5SigmaIEtaIPhi.push_back(sep);
    phoFull5x5E2x5Max   .push_back(lazyToolnoZS-> e2x5Max(theseed) );
    phoFull5x5E2x5Left  .push_back(lazyToolnoZS-> e2x5Left(theseed) );
    phoFull5x5E2x5Right .push_back(lazyToolnoZS-> e2x5Right(theseed) );
    phoFull5x5E2x5Top   .push_back(lazyToolnoZS-> e2x5Top(theseed) );
    phoFull5x5E2x5Bottom.push_back(lazyToolnoZS-> e2x5Bottom(theseed) );
  }
  
  // Cluster shapes
  writeValueMap(iEvent, src, phoFull5x5SigmaIPhiIPhi, phoFull5x5SigmaIPhiIPhi_);  
  writeValueMap(iEvent, src, phoFull5x5SigmaIEtaIPhi, phoFull5x5SigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, phoFull5x5E2x5Max, phoFull5x5E2x5Max_);  
  writeValueMap(iEvent, src, phoFull5x5E2x5Left, phoFull5x5E2x5Left_);  
  writeValueMap(iEvent, src, phoFull5x5E2x5Right, phoFull5x5E2x5Right_);  
  writeValueMap(iEvent, src, phoFull5x5E2x5Top, phoFull5x5E2x5Top_);  
  writeValueMap(iEvent, src, phoFull5x5E2x5Bottom, phoFull5x5E2x5Bottom_);  
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

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

#include <memory>
#include <vector>

class ElectronIDValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronIDValueMapProducer(const edm::ParameterSet&);
  ~ElectronIDValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
		     const std::vector<float> & values,
		     const std::string    & label) const ;
  
  noZS::EcalClusterLazyTools *lazyToolnoZS;

  edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollection_;
  edm::EDGetToken src_;

  std::string dataFormat_;
 
  constexpr static char eleFull5x5SigmaIEtaIEta_[] = "eleFull5x5SigmaIEtaIEta";
  constexpr static char eleFull5x5SigmaIEtaIPhi_[] = "eleFull5x5SigmaIEtaIPhi";
  constexpr static char eleFull5x5E1x5_[] = "eleFull5x5E1x5";
  constexpr static char eleFull5x5E2x5_[] = "eleFull5x5E2x5";
  constexpr static char eleFull5x5E5x5_[] = "eleFull5x5E5x5";
  constexpr static char eleFull5x5R9_[] = "eleFull5x5R9";
  constexpr static char eleFull5x5Circularity_[] = "eleFull5x5Circularity";
};

constexpr char ElectronIDValueMapProducer::eleFull5x5SigmaIEtaIEta_[];
constexpr char ElectronIDValueMapProducer::eleFull5x5SigmaIEtaIPhi_[];
constexpr char ElectronIDValueMapProducer::eleFull5x5E1x5_[];
constexpr char ElectronIDValueMapProducer::eleFull5x5E2x5_[];
constexpr char ElectronIDValueMapProducer::eleFull5x5E5x5_[];
constexpr char ElectronIDValueMapProducer::eleFull5x5R9_[];
constexpr char ElectronIDValueMapProducer::eleFull5x5Circularity_[];

ElectronIDValueMapProducer::ElectronIDValueMapProducer(const edm::ParameterSet& iConfig) {

  ebReducedRecHitCollection_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection"));
  eeReducedRecHitCollection_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection"));
  esReducedRecHitCollection_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("esReducedRecHitCollection"));

  src_ = consumes<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("src"));

  dataFormat_ = iConfig.getParameter<std::string>("dataFormat");
  if( dataFormat_ != "RECO" &&  dataFormat_ != "PAT") {
    throw cms::Exception("InvalidConfiguration") 
      << "ElectronIDValueMapProducer runs in \"RECO\" or \"PAT\" mode!";
  }
  
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

  using namespace edm;
  
  lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_); //, esReducedRecHitCollection_
 
  edm::Handle<edm::View<reco::GsfElectron> > src;
  iEvent.getByToken(src_, src);
  
  if( dataFormat_ == "PAT" && src->size() ) {
    edm::Ptr<pat::Electron> test(src->ptrAt(0));
    if( test.isNull() || !test.isAvailable() ) {
      throw cms::Exception("InvalidConfiguration")
	<<"DataFormat set to \"PAT\" but cannot cast to pat::Electron!";
    }
  }

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
  
  delete lazyToolnoZS;
}

void ElectronIDValueMapProducer::writeValueMap(edm::Event &iEvent,
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

void ElectronIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronIDValueMapProducer);

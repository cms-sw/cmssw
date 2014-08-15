#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

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

class ElectronIDValueMapProducer : public edm::EDProducer {

  public:
  
  explicit ElectronIDValueMapProducer(const edm::ParameterSet&);
  ~ElectronIDValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<reco::GsfElectronCollection> & handle,
		     const std::vector<float> & values,
		     const std::string    & label) const ;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<pat::Electron> > & handle,
		     const std::vector<float> & values,
		     const std::string    & label) const ;
  
  noZS::EcalClusterLazyTools *lazyToolnoZS;

  edm::InputTag ebReducedRecHitCollection_;
  edm::InputTag eeReducedRecHitCollection_;
  edm::InputTag esReducedRecHitCollection_;
  edm::InputTag src_;

  std::string dataFormat_;
};

ElectronIDValueMapProducer::ElectronIDValueMapProducer(const edm::ParameterSet& iConfig) {

  ebReducedRecHitCollection_ = iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection");
  eeReducedRecHitCollection_ = iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection");
  esReducedRecHitCollection_ = iConfig.getParameter<edm::InputTag>("esReducedRecHitCollection");

  src_ = iConfig.getParameter<edm::InputTag>("src");

  dataFormat_ = iConfig.getParameter<std::string>("dataFormat");

  produces<edm::ValueMap<float> >("eleFull5x5SigmaIEtaIEta");  
  produces<edm::ValueMap<float> >("eleFull5x5SigmaIEtaIPhi");  
  produces<edm::ValueMap<float> >("eleFull5x5R9");  
  produces<edm::ValueMap<float> >("eleFull5x5Circularity");  

}

ElectronIDValueMapProducer::~ElectronIDValueMapProducer() {
}

void ElectronIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  
  lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_, esReducedRecHitCollection_);

  if (dataFormat_ == "RECO") {

    edm::Handle<reco::GsfElectronCollection> src;
    iEvent.getByLabel(src_, src);
    
    // size_t n = src->size();
    std::vector<float> eleFull5x5SigmaIEtaIEta, eleFull5x5SigmaIEtaIPhi;
    std::vector<float> eleFull5x5R9, eleFull5x5Circularity;

    for (reco::GsfElectronCollection::const_iterator iEle = src->begin(); iEle != src->end(); ++iEle){

      std::vector<float> vCov = lazyToolnoZS->localCovariances( *(iEle->superCluster()->seed()) );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float sep = vCov[1];
      eleFull5x5SigmaIEtaIEta.push_back(see);
      eleFull5x5SigmaIEtaIPhi.push_back(sep);
      
      eleFull5x5R9.push_back(lazyToolnoZS->e3x3( *((*iEle).superCluster()->seed()) ) / iEle->superCluster()->rawEnergy() );
	
      const float e1x5 = lazyToolnoZS->e5x5( *((*iEle).superCluster()->seed()) );
      const float e5x5 = lazyToolnoZS->e5x5( *((*iEle).superCluster()->seed()) );
      const float circularity = (e5x5 != 0.) ? 1.-e1x5/e5x5 : -1;
      eleFull5x5Circularity.push_back(circularity);
    }
  
    writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIEta, "eleFull5x5SigmaIEtaIEta");  
    writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIPhi, "eleFull5x5SigmaIEtaIPhi");  
    writeValueMap(iEvent, src, eleFull5x5R9, "eleFull5x5R9");  
    writeValueMap(iEvent, src, eleFull5x5Circularity, "eleFull5x5Circularity");  

  } else if (dataFormat_ == "PAT") {

    edm::Handle<edm::View<pat::Electron> > src;
    iEvent.getByLabel(src_, src);
    
    // size_t n = src->size();
    std::vector<float> eleFull5x5SigmaIEtaIEta, eleFull5x5SigmaIEtaIPhi;
    std::vector<float> eleFull5x5R9, eleFull5x5Circularity;
    
    for (View<pat::Electron>::const_iterator iEle = src->begin(); iEle != src->end(); ++iEle) {
      
      std::vector<float> vCov = lazyToolnoZS->localCovariances( *((*iEle).superCluster()->seed()) );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float sep = vCov[1];
      eleFull5x5SigmaIEtaIEta.push_back(see);
      eleFull5x5SigmaIEtaIPhi.push_back(sep);

      eleFull5x5R9.push_back(lazyToolnoZS->e3x3( *((*iEle).superCluster()->seed()) ) / iEle->superCluster()->rawEnergy() );

      const float e1x5 = lazyToolnoZS->e5x5( *((*iEle).superCluster()->seed()) );
      const float e5x5 = lazyToolnoZS->e5x5( *((*iEle).superCluster()->seed()) );
      const float circularity = (e5x5 != 0.) ? 1.-e1x5/e5x5 : -1;
      eleFull5x5Circularity.push_back(circularity);
    }
  
    writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIEta, "eleFull5x5SigmaIEtaIEta");  
    writeValueMap(iEvent, src, eleFull5x5SigmaIEtaIPhi, "eleFull5x5SigmaIEtaIPhi");
    writeValueMap(iEvent, src, eleFull5x5R9, "eleFull5x5R9");  
    writeValueMap(iEvent, src, eleFull5x5Circularity, "eleFull5x5Circularity");  

  }

  delete lazyToolnoZS;
}

void ElectronIDValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<reco::GsfElectronCollection> & handle,
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

void ElectronIDValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<pat::Electron> > & handle,
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

void ElectronIDValueMapProducer::beginJob() {
}

void ElectronIDValueMapProducer::endJob() {
}

void ElectronIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronIDValueMapProducer);

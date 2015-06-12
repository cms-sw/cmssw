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

#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Phys14NonTrig.h"

#include <memory>
#include <vector>

class ElectronMVAValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronMVAValueMapProducer(const edm::ParameterSet&);
  ~ElectronMVAValueMapProducer();
  
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
  
  // for AOD case
  edm::EDGetToken src_;

  // for miniAOD case
  edm::EDGetToken srcMiniAOD_;

  // Electron MVA estimator
  ElectronMVAEstimatorRun2Phys14NonTrig mvaEstimatorPhys14NonTrig_;

  constexpr static char mvaValuesMapName_[] = "eleMVAPhys14NonTrigValues";
  constexpr static char mvaCategoriesMapName_[] = "eleMVAPhys14NonTrigCategories";
};

constexpr char ElectronMVAValueMapProducer::mvaValuesMapName_[];
constexpr char ElectronMVAValueMapProducer::mvaCategoriesMapName_[];

ElectronMVAValueMapProducer::ElectronMVAValueMapProducer(const edm::ParameterSet& iConfig) :
  mvaEstimatorPhys14NonTrig_( iConfig.getParameterSet("mvaConfigPhys14NonTrig") )
{

  //
  // Declare consummables, handle both AOD and miniAOD case
  //
  src_        = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("src"));
  srcMiniAOD_ = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"));

  produces<edm::ValueMap<float> >(mvaValuesMapName_);  
  produces<edm::ValueMap<int> >(mvaCategoriesMapName_);  

}

ElectronMVAValueMapProducer::~ElectronMVAValueMapProducer() {
}

void ElectronMVAValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  
  edm::Handle<edm::View<reco::GsfElectron> > src;

  // Retrieve the collection of electrons from the event.
  // If we fail to retrieve the collection with the standard AOD
  // name, we next look for the one with the stndard miniAOD name.
  iEvent.getByToken(src_, src);
  if( !src.isValid() ){
    iEvent.getByToken(srcMiniAOD_,src);
    if( !src.isValid() )
      throw cms::Exception(" Collection not found: ")
	<< " failed to find a standard AOD or miniAOD electron collection " << std::endl;
  }

 
  // size_t n = src->size();
  std::vector<float> mvaValues;
  std::vector<int> mvaCategories;
  
  // reco::GsfElectron::superCluster() is virtual so we can exploit polymorphism
  for (size_t i = 0; i < src->size(); ++i){
    auto iEle = src->ptrAt(i);

    mvaValues.push_back( mvaEstimatorPhys14NonTrig_.mvaValue(  iEle ) );
    mvaCategories.push_back( mvaEstimatorPhys14NonTrig_.findCategory(  iEle ) );

  }
  
  writeValueMap(iEvent, src, mvaValues, mvaValuesMapName_);  
  writeValueMap(iEvent, src, mvaCategories, mvaCategoriesMapName_);  

}

void ElectronMVAValueMapProducer::writeValueMap(edm::Event &iEvent,
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

void ElectronMVAValueMapProducer::writeValueMap(edm::Event &iEvent,
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

void ElectronMVAValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronMVAValueMapProducer);

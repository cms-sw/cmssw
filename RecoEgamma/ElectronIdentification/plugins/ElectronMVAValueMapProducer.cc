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

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include <memory>
#include <vector>

class ElectronMVAValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronMVAValueMapProducer(const edm::ParameterSet&);
  ~ElectronMVAValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  template<typename T>
  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
		     const std::vector<T> & values,
		     const std::string    & label) const ;
  
  // void writeValueMap(edm::Event &iEvent,
  // 		     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
  // 		     const std::vector<int> & values,
  // 		     const std::string    & label) const ;
  
  // for AOD case
  edm::EDGetToken src_;

  // for miniAOD case
  edm::EDGetToken srcMiniAOD_;

  // MVA estimator
  std::vector<std::unique_ptr<AnyMVAEstimatorRun2Base>> mvaEstimators_;

  // Value map names
  std::vector <std::string> mvaValueMapNames_;
  std::vector <std::string> mvaCategoriesMapNames_;

};

ElectronMVAValueMapProducer::ElectronMVAValueMapProducer(const edm::ParameterSet& iConfig) 
{

  //
  // Declare consummables, handle both AOD and miniAOD case
  //
  src_        = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("src"));
  srcMiniAOD_ = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"));

  // Loop over the list of MVA configurations passed here from python and
  // construct all requested MVA esimtators.
  const std::vector<edm::ParameterSet>& mvaEstimatorConfigs
    = iConfig.getParameterSetVector("mvaConfigurations");
  for( auto &imva : mvaEstimatorConfigs ){

    std::unique_ptr<AnyMVAEstimatorRun2Base> thisEstimator;
    thisEstimator.reset(NULL);
    const edm::ParameterSet& pConf = iConfig.getParameterSet("mvaConfig");
    if( !pConf.empty() ) {
      const std::string& pName = pConf.getParameter<std::string>("mvaName");
      // The factory below constructs the MVA of the appropriate type based
      // on the "mvaName" which is the name of the derived MVA class (plugin)
      AnyMVAEstimatorRun2Base *estimator = AnyMVAEstimatorRun2Factory::get()->create(pName, pConf);
      thisEstimator.reset(estimator);
      
    } else 
      throw cms::Exception(" MVA configuration not found: ")
	<< " failed to find proper configuration for one of the MVAs in the main python script " << std::endl;

    mvaEstimators_.emplace_back( thisEstimator.release() );

    //
    // Compose and save the names of the value maps to be produced
    //
    std::string thisValueMapName = thisEstimator->getName() + "Values";
    std::string thisCategoriesMapName = thisEstimator->getName() + "Categories";
    mvaValueMapNames_.push_back( thisValueMapName );
    mvaCategoriesMapNames_.push_back( thisCategoriesMapName );

    // Declare the maps to the framework
    produces<edm::ValueMap<float> >(thisValueMapName);  
    produces<edm::ValueMap<int> >(thisCategoriesMapName);  

  }


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

 
  // Loop over MVA estimators
  for( unsigned iEstimator = 0; iEstimator < mvaEstimators_.size(); iEstimator++ ){

    std::vector<float> mvaValues;
    std::vector<int> mvaCategories;

    // Loop over particles
    for (size_t i = 0; i < src->size(); ++i){
      auto iEle = src->ptrAt(i);
      
      mvaValues.push_back( mvaEstimators_[iEstimator]->mvaValue(  iEle ) );
      mvaCategories.push_back( mvaEstimators_[iEstimator]->findCategory(  iEle ) );
    } // end loop over particles

    writeValueMap(iEvent, src, mvaValues, mvaValueMapNames_[iEstimator] );  
    writeValueMap(iEvent, src, mvaCategories, mvaCategoriesMapNames_[iEstimator] );  

  } // end loop over estimators
  

}

template<typename T>
void ElectronMVAValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
					     const std::vector<T> & values,
					     const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  auto_ptr<ValueMap<T> > valMap(new ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(valMap, label);
}

// void ElectronMVAValueMapProducer::writeValueMap(edm::Event &iEvent,
// 					     const edm::Handle<edm::View<reco::GsfElectron> > & handle,
// 					     const std::vector<int> & values,
// 					     const std::string    & label) const 
// {
//   using namespace edm; 
//   using namespace std;
//   auto_ptr<ValueMap<int> > valMap(new ValueMap<int>());
//   edm::ValueMap<int>::Filler filler(*valMap);
//   filler.insert(handle, values.begin(), values.end());
//   filler.fill();
//   iEvent.put(valMap, label);
// }

void ElectronMVAValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronMVAValueMapProducer);

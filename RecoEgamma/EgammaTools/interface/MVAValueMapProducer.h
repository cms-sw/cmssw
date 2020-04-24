#ifndef __RecoEgamma_EgammaTools_MVAValueMapProducer_H__
#define __RecoEgamma_EgammaTools_MVAValueMapProducer_H__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "RecoEgamma/EgammaTools/interface/MVAObjectCache.h"

#include <memory>
#include <vector>

template <class ParticleType> 
class MVAValueMapProducer : public edm::stream::EDProducer< edm::GlobalCache<egamma::MVAObjectCache> > {

  public:
  
  MVAValueMapProducer(const edm::ParameterSet&, const egamma::MVAObjectCache*);
  ~MVAValueMapProducer() override;
  
  static std::unique_ptr<egamma::MVAObjectCache>
  initializeGlobalCache(const edm::ParameterSet& conf) {
    return std::make_unique<egamma::MVAObjectCache>(conf);
   }

  static void globalEndJob(const egamma::MVAObjectCache * ) {
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  void produce(edm::Event&, const edm::EventSetup&) override;

  template<typename T>
  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<ParticleType> > & handle,
		     const std::vector<T> & values,
		     const std::string    & label) const ;
  
  // for AOD case
  edm::EDGetToken src_;

  // for miniAOD case
  edm::EDGetToken srcMiniAOD_;

  // MVA estimators are now stored in MVAObjectCache!
  
  // Value map names
  std::vector <std::string> mvaValueMapNames_;
  std::vector <std::string> mvaRawValueMapNames_;
  std::vector <std::string> mvaCategoriesMapNames_;

};

template <class ParticleType>
MVAValueMapProducer<ParticleType>::MVAValueMapProducer(const edm::ParameterSet& iConfig,
                                                       const egamma::MVAObjectCache* mva_cache) 
{

  //
  // Declare consummables, handle both AOD and miniAOD case
  //
  src_        = mayConsume<edm::View<ParticleType> >(iConfig.getParameter<edm::InputTag>("src"));
  srcMiniAOD_ = mayConsume<edm::View<ParticleType> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"));

  // Loop over the list of MVA configurations passed here from python and
  // construct all requested MVA esimtators.  
  const auto& all_mvas = mva_cache->allMVAs();
  for( auto mvaItr = all_mvas.begin(); mvaItr != all_mvas.end(); ++mvaItr ) {
    // set the consumes
    mvaItr->second->setConsumes(consumesCollector());
    //
    // Compose and save the names of the value maps to be produced
    //
    const auto& currentEstimator = mvaItr->second;
    const std::string full_name = ( currentEstimator->getName() + 
                                    currentEstimator->getTag()    );
    std::string thisValueMapName = full_name + "Values";
    std::string thisRawValueMapName = full_name + "RawValues";
    std::string thisCategoriesMapName = full_name + "Categories";    
    mvaValueMapNames_.push_back( thisValueMapName );
    mvaRawValueMapNames_.push_back( thisRawValueMapName );
    mvaCategoriesMapNames_.push_back( thisCategoriesMapName );

    // Declare the maps to the framework
    produces<edm::ValueMap<float> >(thisValueMapName);  
    produces<edm::ValueMap<float> >(thisRawValueMapName);
    produces<edm::ValueMap<int> >(thisCategoriesMapName);
  }


}

template <class ParticleType>
MVAValueMapProducer<ParticleType>::~MVAValueMapProducer() {
}

template <class ParticleType>
void MVAValueMapProducer<ParticleType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  
  edm::Handle<edm::View<ParticleType> > src;

  // Retrieve the collection of particles from the event.
  // If we fail to retrieve the collection with the standard AOD
  // name, we next look for the one with the stndard miniAOD name.
  iEvent.getByToken(src_, src);
  if( !src.isValid() ){
    iEvent.getByToken(srcMiniAOD_,src);
    if( !src.isValid() )
      throw cms::Exception(" Collection not found: ")
	<< " failed to find a standard AOD or miniAOD particle collection " << std::endl;
  }

 
  // Loop over MVA estimators
  const auto& all_mvas = globalCache()->allMVAs();
  for( auto mva_itr = all_mvas.begin(); mva_itr != all_mvas.end(); ++mva_itr ){    
    const int iEstimator = std::distance(all_mvas.begin(),mva_itr);

    // Set up all event content, such as ValueMaps produced upstream or other,
    // original event data pieces, that is needed (if any is implemented in the specific
    // MVA classes)
    const auto& thisEstimator = mva_itr->second;

    std::vector<float> mvaValues;
    std::vector<float> mvaRawValues;
    std::vector<int> mvaCategories;
    
    // Loop over particles
    for (size_t i = 0; i < src->size(); ++i){
      auto iCand = src->ptrAt(i);      
      const float response = thisEstimator->mvaValue( iCand, iEvent );
      mvaRawValues.push_back( response ); // The MVA score
      mvaValues.push_back( 2.0/(1.0+exp(-2.0*response))-1 ); // MVA output between -1 and 1
      mvaCategories.push_back( thisEstimator->findCategory( iCand ) );
    } // end loop over particles

    writeValueMap(iEvent, src, mvaValues, mvaValueMapNames_[iEstimator] );  
    writeValueMap(iEvent, src, mvaRawValues, mvaRawValueMapNames_[iEstimator] );
    writeValueMap(iEvent, src, mvaCategories, mvaCategoriesMapNames_[iEstimator] );
  } // end loop over estimators
  

}

template<class ParticleType> template<typename T>
void MVAValueMapProducer<ParticleType>::writeValueMap(edm::Event &iEvent,
                                                      const edm::Handle<edm::View<ParticleType> > & handle,
                                                      const std::vector<T> & values,
                                                      const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  auto valMap = std::make_unique<ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

template <class ParticleType>
  void MVAValueMapProducer<ParticleType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

#endif

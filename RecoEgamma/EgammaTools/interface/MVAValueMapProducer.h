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

#include "RecoEgamma/EgammaTools/interface/Utils.h"
#include "RecoEgamma/EgammaTools/interface/MultiToken.h"

#include <memory>
#include <vector>
#include <cmath>

template <class ParticleType>
class MVAValueMapProducer : public edm::stream::EDProducer<> {

  public:

  MVAValueMapProducer(const edm::ParameterSet&);
  ~MVAValueMapProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:

  void produce(edm::Event&, const edm::EventSetup&) override;

  // for AOD and MiniAOD case
  MultiTokenT<edm::View<ParticleType>> src_;

  // MVA estimator
  std::vector<std::unique_ptr<AnyMVAEstimatorRun2Base>> mvaEstimators_;

  // Value map names
  std::vector <std::string> mvaValueMapNames_;
  std::vector <std::string> mvaRawValueMapNames_;
  std::vector <std::string> mvaCategoriesMapNames_;

};

template <class ParticleType>
MVAValueMapProducer<ParticleType>::MVAValueMapProducer(const edm::ParameterSet& iConfig)
  : src_(consumesCollector(), iConfig, "src", "srcMiniAOD")

{
  // Loop over the list of MVA configurations passed here from python and
  // construct all requested MVA estimators.
  const std::vector<edm::ParameterSet>& mvaEstimatorConfigs
    = iConfig.getParameterSetVector("mvaConfigurations");

  for( auto &imva : mvaEstimatorConfigs ){

    // The factory below constructs the MVA of the appropriate type based
    // on the "mvaName" which is the name of the derived MVA class (plugin)
    if( !imva.empty() ) {

      mvaEstimators_.emplace_back(AnyMVAEstimatorRun2Factory::get()->create(
                  imva.getParameter<std::string>("mvaName"), imva));

    } else
      throw cms::Exception(" MVA configuration not found: ")
        << " failed to find proper configuration for one of the MVAs in the main python script " << std::endl;

    mvaEstimators_.back()->setConsumes( consumesCollector() );

    //
    // Compose and save the names of the value maps to be produced
    //

    const std::string fullName = ( mvaEstimators_.back()->getName() +
                                   mvaEstimators_.back()->getTag()  );

    const std::string thisValueMapName      = fullName + "Values";
    const std::string thisRawValueMapName   = fullName + "RawValues";
    const std::string thisCategoriesMapName = fullName + "Categories";

    mvaValueMapNames_     .push_back( thisValueMapName      );
    mvaRawValueMapNames_  .push_back( thisRawValueMapName   );
    mvaCategoriesMapNames_.push_back( thisCategoriesMapName );

    // Declare the maps to the framework
    produces<edm::ValueMap<float>>(thisValueMapName     );
    produces<edm::ValueMap<float>>(thisRawValueMapName  );
    produces<edm::ValueMap<int>>  (thisCategoriesMapName);

  }

}

template <class ParticleType>
MVAValueMapProducer<ParticleType>::~MVAValueMapProducer() {
}

template <class ParticleType>
void MVAValueMapProducer<ParticleType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<edm::View<ParticleType> > src = src_.getValidHandle(iEvent);

  // Loop over MVA estimators
  for( unsigned iEstimator = 0; iEstimator < mvaEstimators_.size(); iEstimator++ ){

    std::vector<float> mvaValues;
    std::vector<float> mvaRawValues;
    std::vector<int>   mvaCategories;

    // Loop over particles
    for (size_t i = 0; i < src->size(); ++i){
      auto iCand = src->ptrAt(i);
      int cat = -1; // Passed by reference to the mvaValue function to store the category
      const float response = mvaEstimators_[iEstimator]->mvaValue( iCand, iEvent, cat );
      mvaRawValues.push_back( response ); // The MVA score
      mvaValues.push_back( 2.0/(1.0+exp(-2.0*response))-1 ); // MVA output between -1 and 1
      mvaCategories.push_back( cat );
    } // end loop over particles

    writeValueMap(iEvent, src, mvaValues    , mvaValueMapNames_     [iEstimator] );
    writeValueMap(iEvent, src, mvaRawValues , mvaRawValueMapNames_  [iEstimator] );
    writeValueMap(iEvent, src, mvaCategories, mvaCategoriesMapNames_[iEstimator] );

  } // end loop over estimators

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

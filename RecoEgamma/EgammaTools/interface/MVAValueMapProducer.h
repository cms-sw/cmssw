#ifndef __RecoEgamma_EgammaTools_MVAValueMapProducer_H__
#define __RecoEgamma_EgammaTools_MVAValueMapProducer_H__

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoEgamma/EgammaTools/interface/validateEgammaCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

template <class ParticleType>
class MVAValueMapProducer : public edm::global::EDProducer<> {
public:
  MVAValueMapProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  static auto getMVAEstimators(const edm::VParameterSet& vConfig) {
    std::vector<std::unique_ptr<AnyMVAEstimatorRun2Base>> mvaEstimators;

    // Loop over the list of MVA configurations passed here from python and
    // construct all requested MVA estimators.
    for (auto& imva : vConfig) {
      // The factory below constructs the MVA of the appropriate type based
      // on the "mvaName" which is the name of the derived MVA class (plugin)
      if (!imva.empty()) {
        mvaEstimators.emplace_back(
            AnyMVAEstimatorRun2Factory::get()->create(imva.getParameter<std::string>("mvaName"), imva));

      } else
        throw cms::Exception(" MVA configuration not found: ")
            << " failed to find proper configuration for one of the MVAs in the main python script " << std::endl;
    }

    return mvaEstimators;
  }

  static std::vector<std::string> getValueMapNames(const edm::VParameterSet& vConfig, std::string&& suffix) {
    std::vector<std::string> names;
    for (auto& imva : vConfig) {
      names.push_back(imva.getParameter<std::string>("mvaName") + imva.getParameter<std::string>("mvaTag") + suffix);
    }

    return names;
  }

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<edm::View<ParticleType>> srcToken_;
  const edm::EDGetTokenT<edm::View<ParticleType>> keysForValueMapsToken_;

  // MVA estimators
  const std::vector<std::unique_ptr<AnyMVAEstimatorRun2Base>> mvaEstimators_;

  // Value map names
  const std::vector<std::string> mvaValueMapNames_;
  const std::vector<std::string> mvaRawValueMapNames_;
  const std::vector<std::string> mvaCategoriesMapNames_;

  // To get the auxiliary MVA variables
  const MVAVariableHelper variableHelper_;

  CMS_THREAD_SAFE mutable std::atomic<bool> validated_ = false;
};

namespace {

  template <typename ValueType, class HandleType>
  void writeValueMap(edm::Event& iEvent,
                     const edm::Handle<HandleType>& handle,
                     const std::vector<ValueType>& values,
                     const std::string& label) {
    auto valMap = std::make_unique<edm::ValueMap<ValueType>>();
    typename edm::ValueMap<ValueType>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(std::move(valMap), label);
  }

  template <class ParticleType>
  auto getKeysForValueMapsToken(edm::InputTag const& keysForValueMapsTag, edm::ConsumesCollector&& cc) {
    const bool tagGiven = !keysForValueMapsTag.label().empty();
    return tagGiven ? cc.consumes<edm::View<ParticleType>>(keysForValueMapsTag)
                    : edm::EDGetTokenT<edm::View<ParticleType>>{};
  }

}  // namespace

template <class ParticleType>
MVAValueMapProducer<ParticleType>::MVAValueMapProducer(const edm::ParameterSet& iConfig)
    : srcToken_(consumes<edm::View<ParticleType>>(iConfig.getParameter<edm::InputTag>("src"))),
      keysForValueMapsToken_(getKeysForValueMapsToken<ParticleType>(
          iConfig.getParameter<edm::InputTag>("keysForValueMaps"), consumesCollector())),
      mvaEstimators_(getMVAEstimators(iConfig.getParameterSetVector("mvaConfigurations"))),
      mvaValueMapNames_(getValueMapNames(iConfig.getParameterSetVector("mvaConfigurations"), "Values")),
      mvaRawValueMapNames_(getValueMapNames(iConfig.getParameterSetVector("mvaConfigurations"), "RawValues")),
      mvaCategoriesMapNames_(getValueMapNames(iConfig.getParameterSetVector("mvaConfigurations"), "Categories")),
      variableHelper_(consumesCollector()) {
  for (auto const& name : mvaValueMapNames_)
    produces<edm::ValueMap<float>>(name);
  for (auto const& name : mvaRawValueMapNames_)
    produces<edm::ValueMap<float>>(name);
  for (auto const& name : mvaCategoriesMapNames_)
    produces<edm::ValueMap<int>>(name);
}

template <class ParticleType>
void MVAValueMapProducer<ParticleType>::produce(edm::StreamID,
                                                edm::Event& iEvent,
                                                const edm::EventSetup& iSetup) const {
  std::vector<float> auxVariables = variableHelper_.getAuxVariables(iEvent);

  auto srcHandle = iEvent.getHandle(srcToken_);
  auto keysForValueMapsHandle =
      keysForValueMapsToken_.isUninitialized() ? srcHandle : iEvent.getHandle(keysForValueMapsToken_);

  // check if nothing is wrong with the data format of the candidates
  if (!validated_ && !srcHandle->empty()) {
    egammaTools::validateEgammaCandidate((*srcHandle)[0]);
    validated_ = true;
  }

  // Loop over MVA estimators
  for (unsigned iEstimator = 0; iEstimator < mvaEstimators_.size(); iEstimator++) {
    std::vector<float> mvaValues;
    std::vector<float> mvaRawValues;
    std::vector<int> mvaCategories;

    // Loop over particles
    for (auto const& cand : *srcHandle) {
      int cat = -1;  // Passed by reference to the mvaValue function to store the category
      const float response = mvaEstimators_[iEstimator]->mvaValue(&cand, auxVariables, cat);
      mvaRawValues.push_back(response);                             // The MVA score
      mvaValues.push_back(2.0 / (1.0 + exp(-2.0 * response)) - 1);  // MVA output between -1 and 1
      mvaCategories.push_back(cat);
    }  // end loop over particles

    writeValueMap(iEvent, keysForValueMapsHandle, mvaValues, mvaValueMapNames_[iEstimator]);
    writeValueMap(iEvent, keysForValueMapsHandle, mvaRawValues, mvaRawValueMapNames_[iEstimator]);
    writeValueMap(iEvent, keysForValueMapsHandle, mvaCategories, mvaCategoriesMapNames_[iEstimator]);

  }  // end loop over estimators
}

template <class ParticleType>
void MVAValueMapProducer<ParticleType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", {});
  desc.add<edm::InputTag>("keysForValueMaps", {});
  {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription mvaConfigurations;
    mvaConfigurations.setUnknown();
    desc.addVPSet("mvaConfigurations", mvaConfigurations);
  }
  descriptions.addDefault(desc);
}

#endif

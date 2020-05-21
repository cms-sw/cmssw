#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"

#include <algorithm>

using pat::helper::EfficiencyLoader;

EfficiencyLoader::EfficiencyLoader(const edm::ParameterSet &iConfig, edm::ConsumesCollector &&iC) {
  // Get the names (sorted)
  names_ = iConfig.getParameterNamesForType<edm::InputTag>();
  std::sort(names_.begin(), names_.end());

  // get the InputTags
  for (const auto &name : names_) {
    tokens_.push_back(iC.consumes<edm::ValueMap<pat::LookupTableRecord> >(iConfig.getParameter<edm::InputTag>(name)));
  }

  // prepare the Handles
  handles_.resize(names_.size());
}

void EfficiencyLoader::newEvent(const edm::Event &iEvent) {
  for (size_t i = 0, n = names_.size(); i < n; ++i) {
    iEvent.getByToken(tokens_[i], handles_[i]);
  }
}

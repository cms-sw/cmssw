#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionRcd.h"

#include <algorithm>

using pat::helper::KinResolutionsLoader;

KinResolutionsLoader::KinResolutionsLoader(const edm::ParameterSet &iConfig) {
  // Get the names (sorted)
  patlabels_ = iConfig.getParameterNamesForType<std::string>();

  // get the InputTags
  for (const auto &patlabel : patlabels_) {
    eslabels_.push_back(iConfig.getParameter<std::string>(patlabel));
  }

  // prepare the Handles
  handles_.resize(patlabels_.size());

  // 'default' maps to empty string
  for (auto &patlabel : patlabels_) {
    if (patlabel == "default")
      patlabel = "";
  }
}

void KinResolutionsLoader::newEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  for (size_t i = 0, n = patlabels_.size(); i < n; ++i) {
    iSetup.get<KinematicResolutionRcd>().get(eslabels_[i], handles_[i]);
    handles_[i]->setup(iSetup);
  }
}

void KinResolutionsLoader::fillDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<bool>("addResolutions", false)->setComment("Add resolutions into this PAT Object");
  edm::ParameterSetDescription resolutionPSet;
  resolutionPSet.setAllowAnything();
  iDesc.addOptional("resolutions", resolutionPSet)->setComment("Resolution values to get from EventSetup");
}

#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionRcd.h"

#include <algorithm>

using pat::helper::KinResolutionsLoader;

KinResolutionsLoader::KinResolutionsLoader(const edm::ParameterSet &iConfig, edm::ConsumesCollector iCollector) {
  // Get the names (sorted)
  patlabels_ = iConfig.getParameterNamesForType<std::string>();

  // get the InputTags
  estokens_.reserve(patlabels_.size());
  for (auto const &label : patlabels_) {
    estokens_.emplace_back(iCollector.esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>(label))));
  }
  // prepare the resolutions
  resolutions_.resize(patlabels_.size());

  // 'default' maps to empty string
  for (std::vector<std::string>::iterator it = patlabels_.begin(), ed = patlabels_.end(); it != ed; ++it) {
    if (*it == "default")
      *it = "";
  }
}

void KinResolutionsLoader::newEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  for (size_t i = 0, n = patlabels_.size(); i < n; ++i) {
    resolutions_[i] = &iSetup.getData(estokens_[i]);
  }
}

void KinResolutionsLoader::fillDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<bool>("addResolutions", false)->setComment("Add resolutions into this PAT Object");
  edm::ParameterSetDescription resolutionPSet;
  resolutionPSet.setAllowAnything();
  iDesc.addOptional("resolutions", resolutionPSet)->setComment("Resolution values to get from EventSetup");
}

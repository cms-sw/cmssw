#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include <assert.h>

namespace reco {
  namespace tau {
    // Update our reference to the PFCandidates
    void RecoTauBuilderPlugin::beginEvent() {
      evt()->getByLabel(pfCandSrc_, pfCands_);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
EDM_REGISTER_PLUGINFACTORY(RecoTauBuilderPluginFactory, "RecoTauBuilderPluginFactory");
EDM_REGISTER_PLUGINFACTORY(RecoTauModifierPluginFactory, "RecoTauModifierPluginFactory");
EDM_REGISTER_PLUGINFACTORY(RecoTauCleanerPluginFactory, "RecoTauCleanerPluginFactory");

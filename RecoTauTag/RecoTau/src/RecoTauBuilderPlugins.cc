#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace reco { namespace tau {

// Update our reference to the PFCandidates & PVs
void RecoTauBuilderPlugin::beginEvent() {
  evt()->getByLabel(pfCandSrc_, pfCands_);

  edm::Handle<reco::VertexCollection> pvHandle;
  evt()->getByLabel(pvSrc_, pvHandle);
  // Update the primary vertex
  if (pvHandle->size()) {
    pv_ = reco::VertexRef(pvHandle, 0);
  } else {
    edm::LogError("NoPrimaryVertex") << "No primary vertex found in the event!";
  }
}

}}  // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
EDM_REGISTER_PLUGINFACTORY(RecoTauBuilderPluginFactory,
                           "RecoTauBuilderPluginFactory");
EDM_REGISTER_PLUGINFACTORY(RecoTauModifierPluginFactory,
                           "RecoTauModifierPluginFactory");
EDM_REGISTER_PLUGINFACTORY(RecoTauCleanerPluginFactory,
                           "RecoTauCleanerPluginFactory");

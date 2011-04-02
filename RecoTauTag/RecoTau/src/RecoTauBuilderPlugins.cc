#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace reco { namespace tau {

// Update our reference to the PFCandidates & PVs
void RecoTauBuilderPlugin::beginEvent() {
  evt()->getByLabel(pfCandSrc_, pfCands_);
  evt()->getByLabel(pvSrc_, primaryVertices_);
}

// The primary vertex associated to this jet.
reco::VertexRef
RecoTauBuilderPlugin::primaryVertex(const reco::PFJetRef& jetRef) const {
  // Check to make sure we have something to find
  if (!primaryVertices_.isValid()) {
    edm::LogError("PrimaryVerticesMissing")
      << "The Primary vertex collection does not exist in the event!";
    return reco::VertexRef();
  }
  if (!primaryVertices_->size()) {
    edm::LogError("NoPrimaryVertex")
      << "No primary vertex found in the event!";
    return reco::VertexRef();
  }
  // The default case is the vertex with the largest associated track pt
  reco::VertexRef selectedVertex(primaryVertices_, 0);
  //std::cout << "Getting vertex for jet #" << jetRef.key() << std::endl;
  //std::cout << "Initial vertex @ z = " << selectedVertex->z() << std::endl;
  if (useClosestPV_) {
    //std::cout << "Finding closest..." << std::endl;
    const reco::PFJet& jet = *jetRef;
    selectedVertex = closestVertex(primaryVertices_, jet);
    //std::cout << "Closest is @ z = " << selectedVertex->z() << std::endl;
  }
  return selectedVertex;
}

}}  // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
EDM_REGISTER_PLUGINFACTORY(RecoTauBuilderPluginFactory,
                           "RecoTauBuilderPluginFactory");
EDM_REGISTER_PLUGINFACTORY(RecoTauModifierPluginFactory,
                           "RecoTauModifierPluginFactory");
EDM_REGISTER_PLUGINFACTORY(RecoTauCleanerPluginFactory,
                           "RecoTauCleanerPluginFactory");

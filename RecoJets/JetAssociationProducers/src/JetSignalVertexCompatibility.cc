#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetSignalVertexCompatibilityAlgo.h"

#include "JetSignalVertexCompatibility.h"

using namespace reco;

JetSignalVertexCompatibility::JetSignalVertexCompatibility(const edm::ParameterSet &params)
    : algo(params.getParameter<double>("cut"), params.getParameter<double>("temperature")) {
  jetTracksAssocToken = consumes<JetTracksAssociationCollection>(params.getParameter<edm::InputTag>("jetTracksAssoc"));
  primaryVerticesToken = consumes<VertexCollection>(params.getParameter<edm::InputTag>("primaryVertices"));
  builderToken = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));
  produces<JetFloatAssociation::Container>();
}

JetSignalVertexCompatibility::~JetSignalVertexCompatibility() {}

void JetSignalVertexCompatibility::produce(edm::Event &event, const edm::EventSetup &es) {
  algo.resetEvent(&es.getData(builderToken));

  edm::Handle<JetTracksAssociationCollection> jetTracksAssoc;
  event.getByToken(jetTracksAssocToken, jetTracksAssoc);

  edm::Handle<VertexCollection> primaryVertices;
  event.getByToken(primaryVerticesToken, primaryVertices);

  auto result = std::make_unique<JetFloatAssociation::Container>(jetTracksAssoc->keyProduct());

  for (JetTracksAssociationCollection::const_iterator iter = jetTracksAssoc->begin(); iter != jetTracksAssoc->end();
       ++iter) {
    if (primaryVertices->empty())
      (*result)[iter->first] = -1.;

    const TrackRefVector &tracks = iter->second;
    std::vector<float> compatibility = algo.compatibility(*primaryVertices, tracks);

    // the first vertex is the presumed signal vertex
    (*result)[iter->first] = compatibility[0];
  }

  algo.resetEvent(nullptr);

  event.put(std::move(result));
}

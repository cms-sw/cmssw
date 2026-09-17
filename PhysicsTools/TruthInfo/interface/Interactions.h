// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// The proton-proton interactions overlaid in one event, and the vertex that stands
// for each of them.

#ifndef PhysicsTools_TruthInfo_interface_Interactions_h
#define PhysicsTools_TruthInfo_interface_Interactions_h

#include <cstdint>
#include <optional>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/InteractionId.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"

namespace truth {

  // One interaction of the event.
  struct Interaction {
    // The packed EncodedEventId every particle and vertex of this interaction carries.
    uint64_t eventId = 0;
    // The vertex that stands for the interaction point.
    uint32_t vertexId = 0;
    // The vertex is a stand-in with no usable position, so a distance or a resolution
    // measured against it means nothing. Its constituents still count there.
    bool isPlaceholder = false;

    // The signal is the in-time interaction with index 0. Read it here rather than
    // from the position in the list.
    [[nodiscard]] bool isSignal() const { return isSignalEventId(eventId); }
    [[nodiscard]] int bunchCrossing() const { return bunchCrossingOf(eventId); }
    [[nodiscard]] int eventIndex() const { return eventIndexOf(eventId); }
  };

  // Whether a vertex may stand for an interaction. A vertex that neither merged with a
  // SimVertex nor carries a position is a placeholder, and electing it would count the
  // whole interaction at the origin. Time is part of the test, so a real vertex at the
  // origin with a nonzero time is kept.
  [[nodiscard]] bool usableAsInteractionVertex(VertexData const& vertex);

  // Every interaction of the event, the signal first and the pile-up after it ordered by
  // bunch crossing, then by index inside the crossing. One pass over the vertices when
  // the selection preset built the interaction nodes, one pass over the particles when it
  // did not.
  [[nodiscard]] std::vector<Interaction> interactions(Graph const& graph);

  // The interaction the signal particles come from. Empty when the graph holds none,
  // which is what a pile-up-only sub-graph looks like.
  [[nodiscard]] std::optional<Interaction> signalInteraction(Graph const& graph);

}  // namespace truth

#endif

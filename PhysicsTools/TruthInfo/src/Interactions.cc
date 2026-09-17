// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "PhysicsTools/TruthInfo/interface/Interactions.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>

#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/ParticleData.h"

namespace {

  // The signal first, then the pile-up by bunch crossing and by index inside it.
  bool byArrivalOrder(truth::Interaction const& a, truth::Interaction const& b) {
    return std::make_tuple(!a.isSignal(), a.bunchCrossing(), a.eventIndex()) <
           std::make_tuple(!b.isSignal(), b.bunchCrossing(), b.eventIndex());
  }

}  // namespace

bool truth::usableAsInteractionVertex(truth::VertexData const& vertex) {
  if (vertex.hasSim()) {
    return true;
  }
  auto const& position = vertex.position;
  return position.x() != 0. || position.y() != 0. || position.z() != 0. || position.t() != 0.;
}

std::vector<truth::Interaction> truth::interactions(truth::Graph const& graph) {
  std::vector<Interaction> out;

  // An interaction the graph models gets a VertexRole::Interaction node, built by the
  // selection preset, and THAT is the interaction point: it is not a vertex elected to
  // stand for one.
  for (uint32_t v = 0; v < graph.nVertices(); ++v) {
    auto const& data = graph.vertices()[v];
    if (data.vertexRole() == VertexRole::Interaction) {
      out.push_back(Interaction{data.eventId, v, false});
    }
  }

  if (out.empty()) {
    // No preset ran, so no interaction node exists and the best available answer is an
    // elected stand-in: the lowest-numbered usable production vertex of the interaction,
    // which the build order makes the one where the interaction started.
    std::unordered_map<uint64_t, uint32_t> elected;
    std::unordered_map<uint64_t, uint32_t> placeholderOnly;

    for (uint32_t id = 0; id < graph.nParticles(); ++id) {
      const auto production = Particle(&graph, id).productionVertices();
      if (production.empty()) {
        continue;
      }
      const uint32_t vertexId = production.front().id();
      const uint64_t eventId = graph.particles()[id].eventId;
      auto& target = usableAsInteractionVertex(graph.vertices()[vertexId]) ? elected : placeholderOnly;
      auto [it, inserted] = target.emplace(eventId, vertexId);
      if (!inserted) {
        it->second = std::min(it->second, vertexId);
      }
    }

    out.reserve(elected.size() + placeholderOnly.size());
    for (auto const& [eventId, vertexId] : elected) {
      out.push_back(Interaction{eventId, vertexId, false});
    }
    // An interaction with nothing but placeholders still has to resolve, or every
    // composite object built from its constituents silently matches nothing.
    for (auto const& [eventId, vertexId] : placeholderOnly) {
      if (elected.find(eventId) == elected.end()) {
        out.push_back(Interaction{eventId, vertexId, true});
      }
    }
  }

  std::sort(out.begin(), out.end(), byArrivalOrder);
  return out;
}

std::optional<truth::Interaction> truth::signalInteraction(truth::Graph const& graph) {
  const auto all = interactions(graph);
  // The signal sorts first when it is there at all.
  if (!all.empty() && all.front().isSignal()) {
    return all.front();
  }
  return std::nullopt;
}

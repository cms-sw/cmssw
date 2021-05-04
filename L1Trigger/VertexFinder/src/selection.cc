
#include "L1Trigger/VertexFinder/interface/selection.h"

#include <algorithm>
#include <stdexcept>

namespace l1tVertexFinder {

  const l1t::Vertex& getPrimaryVertex(const std::vector<l1t::Vertex>& aVertexCollection) {
    typedef std::vector<edm::Ptr<l1t::Vertex::Track_t>> Tracks_t;

    return getPrimaryVertex(aVertexCollection, [](const Tracks_t& tracks) -> float {
      float sumPt = 0.0;
      for (auto t : tracks)
        sumPt += t->momentum().transverse();
      return sumPt;
    });
  }

  const l1t::Vertex& getPrimaryVertex(
      const std::vector<l1t::Vertex>& aVertexCollection,
      const std::function<float(const std::vector<edm::Ptr<l1t::Vertex::Track_t>>&)>& aFunction) {
    if (aVertexCollection.empty())
      throw std::invalid_argument("Cannot find primary vertex from empty vertex collection");
    return *std::max_element(aVertexCollection.begin(),
                             aVertexCollection.end(),
                             [aFunction](const l1t::Vertex& v1, const l1t::Vertex& v2) -> bool {
                               return (aFunction(v1.tracks()) < aFunction(v2.tracks()));
                             });
  }

}  // end namespace l1tVertexFinder

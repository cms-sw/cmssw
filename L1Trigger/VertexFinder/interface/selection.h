#ifndef __L1Trigger_VertexFinder_selection_h__
#define __L1Trigger_VertexFinder_selection_h__

#include "DataFormats/L1Trigger/interface/Vertex.h"

#include <vector>

namespace l1tVertexFinder {

  //! Returns primary vertex based on default criterion (max sum pT from all constituent tracks); throws if given empty collection.
  const l1t::Vertex& getPrimaryVertex(const std::vector<l1t::Vertex>& aVertexCollection);

  //! Returns vertex for which parameter 'aFunction' returns the highest value; throws if given empty collection.
  const l1t::Vertex& getPrimaryVertex(
      const std::vector<l1t::Vertex>& aVertexCollection,
      const std::function<float(const std::vector<edm::Ptr<l1t::Vertex::Track_t>>&)>& aFunction);

  // const l1t::Vertex& getPrimaryVertex(const std::vector<l1t::Vertex>& aVertexCollection, const std::function<float (const l1t::Vertex::Track_t&)> aFunction);

}  // end namespace l1tVertexFinder

#endif

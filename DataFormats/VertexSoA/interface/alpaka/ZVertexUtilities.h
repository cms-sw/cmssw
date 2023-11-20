#ifndef DataFormats_Vertex_ZVertexUtilities_h
#define DataFormats_Vertex_ZVertexUtilities_h
#include <alpaka/alpaka.hpp>
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexDefinitions.h"

// Previous ZVertexSoA class methods.
// They operate on View and ConstView of the ZVertexSoA.
namespace zVertex {
  namespace utilities {
    using ZVertexSoA = ZVertexLayout<>;
    using ZVertexSoAView = ZVertexLayout<>::View;
    using ZVertexSoAConstView = ZVertexLayout<>::ConstView;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init(ZVertexSoAView &vertices) { vertices.nvFinal() = 0; }

  }  // namespace utilities
}  // namespace zVertex

#endif

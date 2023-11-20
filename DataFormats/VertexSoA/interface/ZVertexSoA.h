#ifndef DataFormats_Vertex_ZVertexSoA_h
#define DataFormats_Vertex_ZVertexSoA_h

#include <Eigen/Core>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(ZVertexLayout,
                    SOA_COLUMN(int16_t, idv),
                    SOA_COLUMN(float, zv),
                    SOA_COLUMN(float, wv),
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(float, ptv2),
                    SOA_COLUMN(int32_t, ndof),
                    SOA_COLUMN(uint16_t, sortInd),
                    SOA_SCALAR(uint32_t, nvFinal))

// Previous ZVertexSoA class methods.
// They operate on View and ConstView of the ZVertexSoA.
namespace zVertex {
  // Common types for both Host and Device code
  using ZVertexSoA = ZVertexLayout<>;
  using ZVertexSoAView = ZVertexLayout<>::View;
  using ZVertexSoAConstView = ZVertexLayout<>::ConstView;

}  // namespace zVertex

#endif

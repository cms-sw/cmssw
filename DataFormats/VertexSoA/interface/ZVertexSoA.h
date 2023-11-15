#ifndef DataFormats_VertexSoA_interface_ZVertexSoA_h
#define DataFormats_VertexSoA_interface_ZVertexSoA_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  GENERATE_SOA_LAYOUT(ZVertexLayout,
                      SOA_COLUMN(int16_t, idv),
                      SOA_COLUMN(float, zv),
                      SOA_COLUMN(float, wv),
                      SOA_COLUMN(float, chi2),
                      SOA_COLUMN(float, ptv2),
                      SOA_COLUMN(int32_t, ndof),
                      SOA_COLUMN(uint16_t, sortInd),
                      SOA_SCALAR(uint32_t, nvFinal))

  // Common types for both Host and Device code
  using ZVertexSoA = ZVertexLayout<>;
  using ZVertexSoAView = ZVertexSoA::View;
  using ZVertexSoAConstView = ZVertexSoA::ConstView;

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init(ZVertexSoAView &vertices) { vertices.nvFinal() = 0; }

}  // namespace reco

#endif  // DataFormats_VertexSoA_interface_ZVertexSoA_h

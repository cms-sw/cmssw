#ifndef DataFormats_VertexSoA_test_alpaka_ZVertexSoA_test_h
#define DataFormats_VertexSoA_test_alpaka_ZVertexSoA_test_h

#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testZVertexSoAT {

  void runKernels(reco::ZVertexBlocksView zvertex_blocks_view, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testZVertexSoAT

#endif  // DataFormats_VertexSoA_test_alpaka_ZVertexSoA_test_h

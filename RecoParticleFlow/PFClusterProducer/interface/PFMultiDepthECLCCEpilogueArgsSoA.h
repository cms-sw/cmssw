#ifndef RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthECLCCEpilogueArgsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthECLCCEpilogueArgsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  using namespace ::cms::alpakaintrinsics;

  GENERATE_SOA_LAYOUT(PFMultiDepthECLCCEpilogueArgsSoALayout,
                      SOA_COLUMN(uint32_t, ccRHFOffset),
                      SOA_COLUMN(uint32_t, ccRHFSize),
                      SOA_COLUMN(uint32_t, rootMap),
                      SOA_COLUMN(uint32_t, rootLocalMap),
                      SOA_COLUMN(uint32_t, blockRHFOffset),
                      SOA_COLUMN(uint64_t, ccEnergySeed),
                      SOA_SCALAR(int, blockCount),
                      SOA_SCALAR(int, size))

  using PFMultiDepthECLCCEpilogueArgsSoA = PFMultiDepthECLCCEpilogueArgsSoALayout<>;
}  // namespace reco

#endif

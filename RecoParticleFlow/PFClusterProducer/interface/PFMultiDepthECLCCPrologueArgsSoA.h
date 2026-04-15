#ifndef RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthECLCCPrologueArgsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthECLCCPrologueArgsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFMultiDepthECLCCPrologueArgsSoALayout,
                      SOA_COLUMN(uint32_t, ccOffset),
                      SOA_COLUMN(uint32_t, ccLocalOffset),
                      SOA_COLUMN(uint32_t, ccSize),
                      SOA_COLUMN(uint32_t, blockInternCCSize),
                      SOA_SCALAR(int, blockCount),
                      SOA_SCALAR(int, size))

  using PFMultiDepthECLCCPrologueArgsSoA = PFMultiDepthECLCCPrologueArgsSoALayout<>;
}  // namespace reco

#endif

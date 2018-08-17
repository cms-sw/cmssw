#ifndef EventFilter_SiPixelRawToDigi_siPixelRawToClusterHeterogeneousProduct_h
#define EventFilter_SiPixelRawToDigi_siPixelRawToClusterHeterogeneousProduct_h

#include "FWCore/Utilities/interface/typedefs.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

namespace siPixelRawToClusterHeterogeneousProduct {
  using CPUProduct = int; // dummy...

  struct error_obj {
    uint32_t rawId;
    uint32_t word;
    unsigned char errorType;
    unsigned char fedId;

    constexpr
    error_obj(uint32_t a, uint32_t b, unsigned char c, unsigned char d):
      rawId(a),
      word(b),
      errorType(c),
      fedId(d)
    { }
  };

  // FIXME split in two
  struct GPUProduct {
    // Needed for digi and cluster CPU output
    uint32_t const * pdigi_h = nullptr;
    uint32_t const * rawIdArr_h = nullptr;
    int32_t const * clus_h = nullptr;
    uint16_t const * adc_h = nullptr;
    GPU::SimpleVector<error_obj> const * error_h = nullptr;

    GPUProduct const * me_d = nullptr;

    // Needed for GPU rechits
    uint16_t const * xx_d;
    uint16_t const * yy_d;
    uint16_t const * adc_d;
    uint16_t const * moduleInd_d;
    uint32_t const * moduleStart_d;
    int32_t const *  clus_d;
    uint32_t const * clusInModule_d;
    uint32_t const * moduleId_d;

    // originally from rechits
    uint32_t const * clusModuleStart_d;

    uint32_t nDigis;
    uint32_t nModules;
    uint32_t nClusters;
  };

  using HeterogeneousDigiCluster = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}

#endif

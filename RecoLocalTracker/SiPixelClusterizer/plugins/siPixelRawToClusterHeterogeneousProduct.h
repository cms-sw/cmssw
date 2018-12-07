#ifndef EventFilter_SiPixelRawToDigi_siPixelRawToClusterHeterogeneousProduct_h
#define EventFilter_SiPixelRawToDigi_siPixelRawToClusterHeterogeneousProduct_h

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
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
  };

  // FIXME split in two
  struct GPUProduct {
    GPUProduct() = default;
    GPUProduct(const GPUProduct&) = delete;
    GPUProduct& operator=(const GPUProduct&) = delete;
    GPUProduct(GPUProduct&&) = default;
    GPUProduct& operator=(GPUProduct&&) = default;

    GPUProduct(SiPixelDigisCUDA&& digis,
               SiPixelClustersCUDA&& clusters,
               uint32_t ndig, uint32_t nmod, uint32_t nclus):
      digis_d(std::move(digis)), clusters_d(std::move(clusters)),
      nDigis(ndig), nModules(nmod), nClusters(nclus)
    {}

    SiPixelDigisCUDA digis_d;
    SiPixelClustersCUDA clusters_d;

    uint32_t nDigis;
    uint32_t nModules;
    uint32_t nClusters;
  };

  using HeterogeneousDigiCluster = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}

#endif

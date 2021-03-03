#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::maxNumClusters; }
  using hindex_type = uint32_t;  // if above is <=2^32

  using PhiBinner =
      cms::cuda::HistoContainer<int16_t, 128, gpuClustering::maxNumClusters, 8 * sizeof(int16_t), hindex_type, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;

  __device__ __forceinline__ uint32_t nHits() const { return m_nHits; }

  __device__ __forceinline__ float& xLocal(int i) { return m_xl[i]; }
  __device__ __forceinline__ float xLocal(int i) const { return __ldg(m_xl + i); }
  __device__ __forceinline__ float& yLocal(int i) { return m_yl[i]; }
  __device__ __forceinline__ float yLocal(int i) const { return __ldg(m_yl + i); }

  __device__ __forceinline__ float& xerrLocal(int i) { return m_xerr[i]; }
  __device__ __forceinline__ float xerrLocal(int i) const { return __ldg(m_xerr + i); }
  __device__ __forceinline__ float& yerrLocal(int i) { return m_yerr[i]; }
  __device__ __forceinline__ float yerrLocal(int i) const { return __ldg(m_yerr + i); }

  __device__ __forceinline__ float& xGlobal(int i) { return m_xg[i]; }
  __device__ __forceinline__ float xGlobal(int i) const { return __ldg(m_xg + i); }
  __device__ __forceinline__ float& yGlobal(int i) { return m_yg[i]; }
  __device__ __forceinline__ float yGlobal(int i) const { return __ldg(m_yg + i); }
  __device__ __forceinline__ float& zGlobal(int i) { return m_zg[i]; }
  __device__ __forceinline__ float zGlobal(int i) const { return __ldg(m_zg + i); }
  __device__ __forceinline__ float& rGlobal(int i) { return m_rg[i]; }
  __device__ __forceinline__ float rGlobal(int i) const { return __ldg(m_rg + i); }

  __device__ __forceinline__ int16_t& iphi(int i) { return m_iphi[i]; }
  __device__ __forceinline__ int16_t iphi(int i) const { return __ldg(m_iphi + i); }

  __device__ __forceinline__ int32_t& charge(int i) { return m_charge[i]; }
  __device__ __forceinline__ int32_t charge(int i) const { return __ldg(m_charge + i); }
  __device__ __forceinline__ int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  __device__ __forceinline__ int16_t clusterSizeX(int i) const { return __ldg(m_xsize + i); }
  __device__ __forceinline__ int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  __device__ __forceinline__ int16_t clusterSizeY(int i) const { return __ldg(m_ysize + i); }
  __device__ __forceinline__ uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  __device__ __forceinline__ uint16_t detectorIndex(int i) const { return __ldg(m_detInd + i); }

  __device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  __device__ __forceinline__ uint32_t hitsModuleStart(int i) const { return __ldg(m_hitsModuleStart + i); }

  __device__ __forceinline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __device__ __forceinline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __device__ __forceinline__ PhiBinner& phiBinner() { return *m_phiBinner; }
  __device__ __forceinline__ PhiBinner const& phiBinner() const { return *m_phiBinner; }

  __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // local coord
  float *m_xl, *m_yl;
  float *m_xerr, *m_yerr;

  // global coord
  float *m_xg, *m_yg, *m_zg, *m_rg;
  int16_t* m_iphi;

  // cluster properties
  int32_t* m_charge;
  int16_t* m_xsize;
  int16_t* m_ysize;
  uint16_t* m_detInd;

  // supporting objects
  // m_averageGeometry is corrected for beam spot, not sure where to host it otherwise
  AverageGeometry* m_averageGeometry;              // owned by TrackingRecHit2DHeterogeneous
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  PhiBinner* m_phiBinner;

  uint32_t m_nHits;
};

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

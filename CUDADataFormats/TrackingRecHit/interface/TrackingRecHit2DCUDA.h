#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DCUDA_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DCUDA_h

#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"


namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint16_t;  // if above is <=2^16

  using Hist = HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), uint16_t, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  friend class TrackingRecHit2DCUDA;

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

  __device__ __forceinline__ Hist& phiBinner() { return *m_hist; }
  __device__ __forceinline__ Hist const& phiBinner() const { return *m_hist; }

  __device__ __forceinline__ AverageGeometry & averageGeometry() { return *m_averageGeometry; }
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
  AverageGeometry * m_averageGeometry; // owned (corrected for beam spot: not sure where to host it otherwise)
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  Hist* m_hist;

  uint32_t m_nHits;
};

class TrackingRecHit2DCUDA {
public:
  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DCUDA() = default;

  explicit TrackingRecHit2DCUDA(uint32_t nHits,
                                pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                uint32_t const* hitsModuleStart,
                                cuda::stream_t<>& stream);
  ~TrackingRecHit2DCUDA() = default;

  TrackingRecHit2DCUDA(const TrackingRecHit2DCUDA&) = delete;
  TrackingRecHit2DCUDA& operator=(const TrackingRecHit2DCUDA&) = delete;
  TrackingRecHit2DCUDA(TrackingRecHit2DCUDA&&) = default;
  TrackingRecHit2DCUDA& operator=(TrackingRecHit2DCUDA&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_hist; }
  auto iphi() { return m_iphi; }

  // only the local coord and detector index
  cudautils::host::unique_ptr<float[]> localCoordToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cuda::stream_t<>& stream) const;

private:
  static constexpr uint32_t n16 = 4;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  cudautils::device::unique_ptr<uint16_t[]> m_store16;
  cudautils::device::unique_ptr<float[]> m_store32;

  cudautils::device::unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;
  cudautils::device::unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;

  cudautils::device::unique_ptr<TrackingRecHit2DSOAView> m_view;

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Hist* m_hist;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DCUDA_h

#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h

#include <cstddef>
#include <cstdint>

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"

namespace gpuVertexFinder {

  // SOA for vertices
  // These vertices are clusterized and fitted only along the beam line (z)
  // to obtain their global coordinate the beam spot position shall be added (eventually correcting for the beam angle as well)
  // FIXME move to DataFormats
  struct ZVertices {
    static constexpr uint32_t MAXTRACKS = 16000;
    static constexpr uint32_t MAXVTX = 1024;

    int32_t idv[MAXTRACKS];    // vertex index for each associated (original) track
    float zv[MAXVTX];          // output z-posistion of found vertices
    float wv[MAXVTX];          //  output weight (1/error^2) on the above
    float chi2[MAXVTX];        // vertices chi2
    float ptv2[MAXVTX];        // vertices pt^2
    int32_t ndof[MAXVTX];      // vertices number of dof (resued as workspace for the number of nearest neighbours)
    uint16_t sortInd[MAXVTX];  // sorted index (by pt2)
    uint32_t nvFinal;          // the number of vertices

    __host__ __device__ void init() { nvFinal = 0; }
  };

  // workspace used in the vertex reco algos
  struct WorkSpace {
    static constexpr uint32_t MAXTRACKS = 16000;
    static constexpr uint32_t MAXVTX = 1024;

    uint32_t ntrks;            // number of "selected tracks"
    uint16_t itrk[MAXTRACKS];  // index of original track
    float zt[MAXTRACKS];       // input track z at bs
    float ezt2[MAXTRACKS];     // input error^2 on the above
    float ptt2[MAXTRACKS];     // input pt^2 on the above
    uint8_t izt[MAXTRACKS];    // interized z-position of input tracks
    int32_t iv[MAXTRACKS];     // vertex index for each associated track

    uint32_t nvIntermediate;  // the number of vertices after splitting pruning etc.

    __host__ __device__ void init() {
      ntrks = 0;
      nvIntermediate = 0;
    }
  };

#ifdef __CUDACC__
  __global__ void init(ZVertices* pdata, WorkSpace* pws) {
    pdata->init();
    pws->init();
  }
#endif

  // Data Format on cpu???
  struct OnCPU {
    OnCPU() = default;

    std::vector<float, CUDAHostAllocator<float>> z, zerr, chi2;
    std::vector<int16_t, CUDAHostAllocator<int16_t>> sortInd;
    std::vector<int32_t, CUDAHostAllocator<int32_t>> ivtx;
    std::vector<uint16_t, CUDAHostAllocator<uint16_t>> itrk;

    uint32_t nVertices = 0;
    uint32_t nTracks = 0;
    ZVertices const* gpu_d = nullptr;
  };

  class Producer {
  public:
    using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;

    using OnCPU = gpuVertexFinder::OnCPU;
    using ZVertices = gpuVertexFinder::ZVertices;
    using WorkSpace = gpuVertexFinder::WorkSpace;

    Producer(bool useDensity,
             bool useDBSCAN,
             bool useIterative,
             int iminT,       // min number of neighbours to be "core"
             float ieps,      // max absolute distance to cluster
             float ierrmax,   // max error to be "seed"
             float ichi2max,  // max normalized distance to cluster
             bool ienableTransfer)
        : useDensity_(useDensity),
          useDBSCAN_(useDBSCAN),
          useIterative_(useIterative),
          minT(iminT),
          eps(ieps),
          errmax(ierrmax),
          chi2max(ichi2max),
          enableTransfer(ienableTransfer) {}

    ~Producer() { deallocate(); }

    void produce(cudaStream_t stream, TuplesOnCPU const& tuples, float ptMin);

    OnCPU const& fillResults(cudaStream_t stream);

    void allocate();
    void deallocate();

  private:
    OnCPU gpuProduct;
    ZVertices* gpu_d = nullptr;
    WorkSpace* ws_d = nullptr;

    const bool useDensity_;
    const bool useDBSCAN_;
    const bool useIterative_;

    int minT;       // min number of neighbours to be "core"
    float eps;      // max absolute distance to cluster
    float errmax;   // max error to be "seed"
    float chi2max;  // max normalized distance to cluster
    const bool enableTransfer;
  };

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h

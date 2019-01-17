#ifndef RecoPixelVertexing_PixelVertexFinding_gpuVertexFinder_H
#define RecoPixelVertexing_PixelVertexFinding_gpuVertexFinder_H

#include<cstdint>

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"


namespace gpuVertexFinder {

  struct OnGPU {

    static constexpr uint32_t MAXTRACKS = 16000;
    static constexpr uint32_t MAXVTX= 1024;

    OnGPU() = default;
    OnGPU(nullptr_t):
      ntrks{nullptr}, itrk{nullptr}, zt{nullptr}, ezt2{nullptr}, ptt2{nullptr},
      zv{nullptr}, wv{nullptr}, chi2{nullptr}, ptv2{nullptr}, nvFinal{nullptr},
      nvIntermediate{nullptr}, iv{nullptr}, sortInd{nullptr},
      izt{nullptr}, nn{nullptr}
    {}

    uint32_t * ntrks; // number of "selected tracks"
    uint16_t * itrk; // index of original track    
    float * zt;   // input track z at bs
    float * ezt2; // input error^2 on the above
    float * ptt2; // input pt^2 on the above
   

    float * zv;  // output z-posistion of found vertices
    float * wv;  //  output weight (1/error^2) on the above
    float * chi2;  // vertices chi2
    float * ptv2;  // vertices pt^2
    uint32_t * nvFinal;  // the number of vertices
    uint32_t * nvIntermediate;  // the number of vertices after splitting pruning etc.
    int32_t * iv;  // vertex index for each associated track
    uint16_t * sortInd; // sorted index (by pt2)

    // workspace  
    uint8_t * izt;  // interized z-position of input tracks
    int32_t * nn; // number of nearest neighbours (reused as number of dof for output vertices)
    
  };
  

  struct OnCPU {
    OnCPU() = default;

    std::vector<float,    CUDAHostAllocator<float>> z,zerr, chi2;
    std::vector<int16_t, CUDAHostAllocator<uint16_t>> sortInd;
    std::vector<int32_t, CUDAHostAllocator<int32_t>> ivtx;
    std::vector<uint16_t, CUDAHostAllocator<uint16_t>> itrk;

    uint32_t nVertices=0;
    uint32_t nTracks=0;
    OnGPU const * gpu_d = nullptr;
  };

  class Producer {
  public:

    using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;

    using OnCPU = gpuVertexFinder::OnCPU;
    using OnGPU = gpuVertexFinder::OnGPU;


    Producer(
	     int iminT,  // min number of neighbours to be "core"
	     float ieps, // max absolute distance to cluster
	     float ierrmax, // max error to be "seed"
	     float ichi2max,   // max normalized distance to cluster
             bool ienableTransfer
	     ) :
      onGPU(nullptr),
      minT(iminT),
      eps(ieps),
      errmax(ierrmax),
      chi2max(ichi2max),
      enableTransfer(ienableTransfer)
    {}
    
    ~Producer() { deallocateOnGPU();}

    void produce(cudaStream_t stream, TuplesOnCPU const & tuples, float ptMin);

    OnCPU const & fillResults(cudaStream_t stream);
    

    void allocateOnGPU();
    void deallocateOnGPU();

  private:
    OnCPU gpuProduct;
    OnGPU onGPU;
    OnGPU * onGPU_d=nullptr;

    int minT;  // min number of neighbours to be "core"
    float eps; // max absolute distance to cluster
    float errmax; // max error to be "seed"
    float chi2max;   // max normalized distance to cluster
    const bool enableTransfer;

  };
  
} // end namespace

#endif

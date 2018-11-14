#ifndef RecoPixelVertexing_PixelVertexFinding_gpuVertexFinder_H
#define RecoPixelVertexing_PixelVertexFinding_gpuVertexFinder_H

#include<cstdint>

#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"


namespace gpuVertexFinder {

  struct OnGPU {

    static constexpr uint32_t MAXTRACKS = 16000;
    static constexpr uint32_t MAXVTX= 1024;
    
    float * zt;   // input track z at bs
    float * ezt2; // input error^2 on the above
    float * ptt2; // input pt^2 on the above
   

    float * zv;  // output z-posistion of found vertices
    float * wv;  //  output weight (1/error^2) on the above
    float * chi2;  // vertices chi2
    float * ptv2;  // vertices pt^2
    uint32_t * nv;  // the number of vertices
    int32_t * iv;  // vertex index for each associated track
    uint16_t * sortInd; // sorted index (by pt2)

    // workspace  
    uint8_t * izt;  // interized z-position of input tracks
    int32_t * nn; // number of nearest neighbours (reused as number of dof for output vertices)
    
  };
  

  class Producer {
  public:
    
    using GPUProduct = pixelVertexHeterogeneousProduct::GPUProduct;
    using OnGPU = gpuVertexFinder::OnGPU;


    Producer(
	     int iminT,  // min number of neighbours to be "core"
	     float ieps, // max absolute distance to cluster
	     float ierrmax, // max error to be "seed"
	     float ichi2max   // max normalized distance to cluster
	     ) :
      minT(iminT),
      eps(ieps),
      errmax(ierrmax),
      chi2max(ichi2max)  
    {}
    
    ~Producer() { deallocateOnGPU();}
    
    void produce(cudaStream_t stream,
		 float const * zt,
		 float const * ezt2,
                 float const * ptt2,
		 uint32_t ntrks
		 );
    
    GPUProduct const & fillResults(cudaStream_t stream);
    

    void allocateOnGPU();
    void deallocateOnGPU();

  private:
    GPUProduct gpuProduct;
    OnGPU onGPU;
    OnGPU * onGPU_d=nullptr;

    int minT;  // min number of neighbours to be "core"
    float eps; // max absolute distance to cluster
    float errmax; // max error to be "seed"
    float chi2max;   // max normalized distance to cluster

  };
  
} // end namespace

#endif

#include "gpuClusterTracks.h"

namespace gpuVertexFinder {


  void Producer::allocateOnGPU() {
    cudaCheck(cudaMalloc(&onGPU.zt, OnGPU::MAXTRACKS*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.ezt2, OnGPU::MAXTRACKS*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.ptt2, OnGPU::MAXTRACKS*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.iv, OnGPU::MAXTRACKS*sizeof(int32_t)));

    cudaCheck(cudaMalloc(&onGPU.nv, sizeof(uint32_t)));
    cudaCheck(cudaMalloc(&onGPU.zv, OnGPU::MAXVTX*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.wv, OnGPU::MAXVTX*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.chi2, OnGPU::MAXVTX*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.ptv2, OnGPU::MAXVTX*sizeof(float)));
    cudaCheck(cudaMalloc(&onGPU.sortInd, OnGPU::MAXVTX*sizeof(uint16_t)));


    cudaCheck(cudaMalloc(&onGPU.izt, OnGPU::MAXTRACKS*sizeof(uint8_t)));
    cudaCheck(cudaMalloc(&onGPU.nn, OnGPU::MAXTRACKS*sizeof(int32_t)));

    cudaCheck(cudaMalloc(&onGPU_d,sizeof(OnGPU)));
    cudaCheck(cudaMemcpy(onGPU_d,&onGPU,sizeof(OnGPU),cudaMemcpyHostToDevice));

  }
	      
  void Producer::deallocateOnGPU() {
    cudaCheck(cudaFree(onGPU.zt));
    cudaCheck(cudaFree(onGPU.ezt2));
    cudaCheck(cudaFree(onGPU.ptt2));
    cudaCheck(cudaFree(onGPU.iv));

    cudaCheck(cudaFree(onGPU.nv));
    cudaCheck(cudaFree(onGPU.zv));
    cudaCheck(cudaFree(onGPU.wv));
    cudaCheck(cudaFree(onGPU.chi2));
    cudaCheck(cudaFree(onGPU.ptv2));
    cudaCheck(cudaFree(onGPU.sortInd));

    cudaCheck(cudaFree(onGPU.izt));
    cudaCheck(cudaFree(onGPU.nn));

    cudaCheck(cudaFree(onGPU_d));

  }


  void Producer::produce(cudaStream_t stream,
			 float const * __restrict__ zt,
			 float const * __restrict__ ezt2,
                         float const * __restrict__ ptt2,
			 uint32_t ntrks
			 ) {
    
    cudaCheck(cudaMemcpyAsync(onGPU.zt,zt,ntrks*sizeof(float),
			      cudaMemcpyHostToDevice,stream));
    cudaCheck(cudaMemcpyAsync(onGPU.ezt2,ezt2,ntrks*sizeof(float),
			      cudaMemcpyHostToDevice,stream));
    cudaCheck(cudaMemcpyAsync(onGPU.ptt2,ptt2,ntrks*sizeof(float),
                              cudaMemcpyHostToDevice,stream));
    
    assert(onGPU_d);
    clusterTracks<<<1,1024-256,0,stream>>>(ntrks,onGPU_d,minT,eps,errmax,chi2max);
    cudaCheck(cudaGetLastError());
    sortByPt2<<<1,256,0,stream>>>(ntrks,onGPU_d);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpyAsync(&gpuProduct.nVertices, onGPU.nv, sizeof(uint32_t),
			      cudaMemcpyDeviceToHost, stream));
    
    gpuProduct.ivtx.resize(ntrks);
    cudaCheck(cudaMemcpyAsync(gpuProduct.ivtx.data(),onGPU.iv,sizeof(int32_t)*ntrks,
			      cudaMemcpyDeviceToHost, stream));
  }
  
  Producer::GPUProduct const & Producer::fillResults(cudaStream_t stream) {

    // finish copy
    gpuProduct.z.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.z.data(),onGPU.zv,sizeof(float)*gpuProduct.nVertices,
			      cudaMemcpyDeviceToHost, stream));
    gpuProduct.zerr.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.zerr.data(),onGPU.wv,sizeof(float)*gpuProduct.nVertices,
			      cudaMemcpyDeviceToHost, stream));
    gpuProduct.chi2.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.chi2.data(),onGPU.chi2,sizeof(float)*gpuProduct.nVertices,
			      cudaMemcpyDeviceToHost, stream));
        
    gpuProduct.sortInd.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.sortInd.data(),onGPU.sortInd,sizeof(uint16_t)*gpuProduct.nVertices,
                              cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    
    return gpuProduct;
  }
	
} // end namespace

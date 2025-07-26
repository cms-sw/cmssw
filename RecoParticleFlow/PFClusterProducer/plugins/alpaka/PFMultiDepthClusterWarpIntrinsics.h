#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"


namespace cms::alpakatools{
  namespace warp {

      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void syncWarpThreads_mask(TAcc const& acc, unsigned mask) {
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        __syncwarp(mask); // Synchronize all threads within a subset of lanes in the warp
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        __builtin_amdgcn_wave_barrier();
#endif
#endif	
        // No-op for CPU accelerators 
      } 

      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned ballot_mask(TAcc const& acc, unsigned mask, int pred ) {
        unsigned res{0};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)	
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        res = __ballot_sync(mask, pred); // Synchronize all threads within a warp
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        // HIP equivalent for warp ballot
#endif
#endif	
        return res;
      }

      template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T shfl_mask(TAcc const& acc, unsigned mask, T var, int srcLane, int width ) {
        T res{};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)	
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        res = __shfl_sync(mask, var, srcLane, width); // Synchronize all threads within a warp
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        // HIP equivalent for warp __shfl_down_sync
#endif
#endif	
        return res;
      } 

      template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T shfl_down_mask(TAcc const& acc, unsigned mask, T var, int srcLane, int width ) {
        T res{};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)	
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        res = __shfl_down_sync(mask, var, srcLane, width); // Synchronize all threads within a warp
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        // HIP equivalent for warp __shfl_down_sync
#endif
#endif	
        return res;
      } 

      template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T shfl_up_mask(TAcc const& acc, unsigned mask, T var, int srcLane, int width ) {
        T res{};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)	
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
        res = __shfl_up_sync(mask, var, srcLane, width); // Synchronize all threads within a warp
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        // HIP equivalent for warp __shfl_up_sync
#endif
#endif	
        return res;
      } 

      template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T match_any_mask(TAcc const& acc, unsigned mask, T val) {
        T res{};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)	
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        // Alpaka CUDA backend
#if __CUDA_ARCH__ >= 700
        res = __match_any_sync(mask, val); // Synchronize all threads within a warp
#else
        const unsigned int w_extent = alpaka::warp::getSize(acc);
        unsigned int match = 0;
	for (int iter_lane_idx = 0; iter_lane_idx < w_extent; ++iter_lane_idx) {
          T iter_val = __shfl_sync(mask, val, iter_lane_idx, w_extent);
	  const unsigned int iter_lane_mask = 1 << iter_lane_idx;
          if (iter_val == val) match |= iter_lane_mask;
    	}
    	res = match & mask;

        __syncwarp(mask);
#endif
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        // Alpaka HIP backend
        // HIP equivalent for warp __match_any_sync
#endif
#endif	
        return res;
      } 

  } // end of warp exp

    // reverse the bit order of a (32-bit) unsigned integer.
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned brev(TAcc const& acc, unsigned mask) {
      unsigned res{0};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)      
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // Alpaka CUDA backend
      res = __brev(mask); 
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      // Alpaka HIP backend
#endif
#endif      
      return res;
    }

    // count the number of leading zeros in a 32-bit unsigned integer
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned clz(TAcc const& acc, unsigned mask) {
      unsigned res{0};
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)      
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // Alpaka CUDA backend
      res = __clz(mask); 
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      // Alpaka HIP backend
#endif
#endif      
      return res;
    }     
    
}// end of alpakatools
#endif

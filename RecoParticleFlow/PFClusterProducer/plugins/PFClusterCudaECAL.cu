#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Eigen include files
#include <Eigen/Dense>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "PFClusterCudaECAL.h"

using PFClustering::common::PFLayer;

// Uncomment for debugging
//#define DEBUG_GPU_ECAL

constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);
constexpr const float PI_F = 3.141592654f;
constexpr const float preshowerStartEta = 1.653;
constexpr const float preshowerEndEta = 2.6;

namespace PFClusterCudaECAL {

  __constant__ float showerSigma2;
  __constant__ float recHitEnergyNormInvEB;
  __constant__ float recHitEnergyNormInvEE;
  __constant__ float minFracToKeep;
  __constant__ float minFracTot;
  __constant__ float stoppingTolerance;

  __constant__ float seedEThresholdEB;
  __constant__ float seedEThresholdEE;
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB;
  __constant__ float topoEThresholdEE;

  __constant__ int maxIterations;
  __constant__ bool excludeOtherSeeds;

  __constant__ int nNeigh;
  __constant__ int maxSize;

  // Generic position calc constants
  __constant__ float minAllowedNormalization;
  __constant__ float logWeightDenominatorInv;
  __constant__ float minFractionInCalc;

  // Convergence position calc constants
  __constant__ float conv_minAllowedNormalization;
  __constant__ float conv_T0_ES;
  __constant__ float conv_T0_EE;
  __constant__ float conv_T0_EB;
  __constant__ float conv_X0;
  __constant__ float conv_minFractionInCalc;
  __constant__ float conv_W0;

  int nTopoLoops = 18;  // DEPRECATED: Number of iterations for topo kernel

  //  void initializeCudaConstants(const float h_showerSigma2,
  //                               const float h_recHitEnergyNormInvEB,
  //                               const float h_recHitEnergyNormInvEE,
  //                               const float h_minFracToKeep,
  //                               const float h_minFracTot,
  //                               const uint32_t   h_maxIterations,
  //                               const float h_stoppingTolerance,
  //                               const bool  h_excludeOtherSeeds,
  //                               const float h_seedEThresholdEB,
  //                               const float h_seedEThresholdEE,
  //                               const float h_seedPt2ThresholdEB,
  //                               const float h_seedPt2ThresholdEE,
  //                               const float h_topoEThresholdEB,
  //                               const float h_topoEThresholdEE,
  //                               const int   h_nNeigh,
  //                               const PFClustering::common::PosCalcConfig h_posCalcConfig,
  //                               const PFClustering::common::ECALPosDepthCalcConfig h_convergencePosCalcConfig,
  //                               cudaStream_t cudaStream
  //                           )
  void initializeCudaConstants(const PFClustering::common::CudaECALConstants& cudaConstants,
                               const cudaStream_t cudaStream) {
    cudaCheck(cudaMemcpyToSymbolAsync(
        showerSigma2, &cudaConstants.showerSigma2, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    std::cout << "--- ECAL Cuda constant values ---" << std::endl;
    float val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, showerSigma2, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "showerSigma2 read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormInvEB,
                                      &cudaConstants.recHitEnergyNormInvEB,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(
        cudaMemcpyFromSymbolAsync(&val, recHitEnergyNormInvEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "recHitEnergyNormInvEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormInvEE,
                                      &cudaConstants.recHitEnergyNormInvEE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(
        cudaMemcpyFromSymbolAsync(&val, recHitEnergyNormInvEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "recHitEnergyNormInvEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        minFracToKeep, &cudaConstants.minFracToKeep, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, minFracToKeep, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minFracToKeep read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        minFracTot, &cudaConstants.minFracTot, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, minFracTot, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minFracTot read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        stoppingTolerance, &cudaConstants.stoppingTolerance, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, stoppingTolerance, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "stoppingTolerance read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        excludeOtherSeeds, &cudaConstants.excludeOtherSeeds, sizeof(bool), 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    bool bval = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&bval, excludeOtherSeeds, sizeof(bool), 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "excludeOtherSeeds read from symbol: " << bval << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        maxIterations, &cudaConstants.maxIterations, sizeof_int, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    uint32_t uival = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&uival, maxIterations, sizeof_int, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "maxIterations read from symbol: " << uival << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        seedEThresholdEB, &cudaConstants.seedEThresholdEB, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, seedEThresholdEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedEThresholdEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        seedEThresholdEE, &cudaConstants.seedEThresholdEE, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, seedEThresholdEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedEThresholdEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        seedPt2ThresholdEB, &cudaConstants.seedPt2ThresholdEB, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, seedPt2ThresholdEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedPt2ThresholdEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        seedPt2ThresholdEE, &cudaConstants.seedPt2ThresholdEE, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, seedPt2ThresholdEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedPt2ThresholdEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        topoEThresholdEB, &cudaConstants.topoEThresholdEB, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, topoEThresholdEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "topoEThresholdEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        topoEThresholdEE, &cudaConstants.topoEThresholdEE, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, topoEThresholdEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "topoEThresholdEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(
        cudaMemcpyToSymbolAsync(nNeigh, &cudaConstants.nNeigh, sizeof_int, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    int ival = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&ival, nNeigh, sizeof_int, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "nNeigh read from symbol: " << ival << std::endl;
#endif

    // Generic position calc constants
    cudaCheck(cudaMemcpyToSymbolAsync(minAllowedNormalization,
                                      &cudaConstants.posCalcConfig.minAllowedNormalization,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(
        cudaMemcpyFromSymbolAsync(&val, minAllowedNormalization, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minAllowedNormalization read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(logWeightDenominatorInv,
                                      &cudaConstants.posCalcConfig.logWeightDenominatorInv,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(
        cudaMemcpyFromSymbolAsync(&val, logWeightDenominatorInv, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "logWeightDenominatorInv read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(minFractionInCalc,
                                      &cudaConstants.posCalcConfig.minFractionInCalc,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, minFractionInCalc, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minFractionInCalc read from symbol: " << val << std::endl;
#endif

    // Convergence position calc constants
    cudaCheck(cudaMemcpyToSymbolAsync(conv_minAllowedNormalization,
                                      &cudaConstants.convergencePosCalcConfig.minAllowedNormalization,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val, conv_minAllowedNormalization, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_minAllowedNormalization read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        conv_T0_ES, &cudaConstants.convergencePosCalcConfig.T0_ES, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, conv_T0_ES, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_T0_ES read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        conv_T0_EE, &cudaConstants.convergencePosCalcConfig.T0_EE, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, conv_T0_EE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_T0_EE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        conv_T0_EB, &cudaConstants.convergencePosCalcConfig.T0_EB, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, conv_T0_EB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_T0_EB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        conv_X0, &cudaConstants.convergencePosCalcConfig.X0, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, conv_X0, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_X0 read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(conv_minFractionInCalc,
                                      &cudaConstants.convergencePosCalcConfig.minFractionInCalc,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(
        cudaMemcpyFromSymbolAsync(&val, conv_minFractionInCalc, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_minFractionInCalc read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        conv_W0, &cudaConstants.convergencePosCalcConfig.W0, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_ECAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, conv_W0, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "conv_W0 read from symbol: " << val << std::endl;
#endif
  }

  __device__ __forceinline__ float mag(float xpos, float ypos, float zpos) {
    return sqrtf(xpos * xpos + ypos * ypos + zpos * zpos);
  }

  __device__ __forceinline__ float mag(float4 pos) { return sqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z); }

  __device__ __forceinline__ float phiFromCartesian(float4 pos) { return atan2f(pos.y, pos.x); }

  __device__ __forceinline__ float etaFromCartesian(float posX, float posY, float posZ) {
    float m = mag(posX, posY, posZ);
    float cosTheta = m > 0.0 ? posZ / m : 1.0;
    return (0.5 * logf((1.0 + cosTheta) / (1.0 - cosTheta)));
  }

  __device__ __forceinline__ float etaFromCartesian(float4 pos) {
    float m = mag(pos);
    float cosTheta = m > 0.0 ? pos.z / m : 1.0;
    return (0.5 * logf((1.0 + cosTheta) / (1.0 - cosTheta)));
  }

  __device__ __forceinline__ float dR2(float4 pos1, float4 pos2) {
    float eta1 = etaFromCartesian(pos1);
    float phi1 = phiFromCartesian(pos1);

    float eta2 = etaFromCartesian(pos2);
    float phi2 = phiFromCartesian(pos2);

    float deta = eta2 - eta1;
    float dphi = fabsf(fabsf(phi2 - phi1) - PI_F) - PI_F;
    return (deta * deta + dphi * dphi);
  }

  __global__ void seedingTopoThreshKernel_ECAL(size_t size,
                                               float* fracSum,
                                               const float* __restrict__ pfrh_energy,
                                               const float* __restrict__ pfrh_pt2,
                                               int* pfrh_isSeed,
                                               int* pfrh_topoId,
                                               bool* pfrh_passTopoThresh,
                                               const int* __restrict__ pfrh_layer,
                                               const int* __restrict__ neigh8_Ind,
                                               int* rhCount,
                                               int* topoSeedCount,
                                               int* topoRHCount,
                                               int* seedFracOffsets,
                                               int* topoSeedOffsets,
                                               int* topoSeedList,
                                               int* pfcIter) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size) {
      // Initialize arrays
      pfrh_topoId[i] = i;
      pfrh_isSeed[i] = 0;
      rhCount[i] = 0;
      fracSum[i] = 0.;

      topoSeedCount[i] = 0;
      topoRHCount[i] = 0;
      seedFracOffsets[i] = -1;
      topoSeedOffsets[i] = -1;
      topoSeedList[i] = -1;
      pfcIter[i] = -1;

      // Seeding threshold test
      if ((pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i] > seedEThresholdEB &&
           pfrh_pt2[i] > seedPt2ThresholdEB) ||
          (pfrh_layer[i] == PFLayer::ECAL_ENDCAP && pfrh_energy[i] > seedEThresholdEE &&
           pfrh_pt2[i] > seedPt2ThresholdEE)) {
        pfrh_isSeed[i] = 1;
        for (int k = 0; k < nNeigh; k++) {
          if (neigh8_Ind[nNeigh * i + k] < 0)
            continue;
          if (pfrh_energy[i] < pfrh_energy[neigh8_Ind[nNeigh * i + k]]) {
            pfrh_isSeed[i] = 0;
            //pfrh_topoId[i]=-1;
            break;
          }
        }
      } else {
        //pfrh_topoId[i]=-1;
        pfrh_isSeed[i] = 0;
      }

      // Topo clustering threshold test
      if ((pfrh_layer[i] == PFLayer::ECAL_ENDCAP && pfrh_energy[i] > topoEThresholdEE) ||
          (pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i] > topoEThresholdEB)) {
        pfrh_passTopoThresh[i] = true;
      }
      //else { pfrh_passTopoThresh[i] = false; }
      else {
        pfrh_passTopoThresh[i] = false;
        pfrh_topoId[i] = -1;
      }
    }
  }

  __global__ void seedingKernel_ECAL(size_t size,
                                     const float* __restrict__ pfrh_energy,
                                     const float* __restrict__ pfrh_pt2,
                                     int* pfrh_isSeed,
                                     int* pfrh_topoId,
                                     const int* __restrict__ pfrh_layer,
                                     const int* __restrict__ neigh8_Ind) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size) {
      if ((pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i] > seedEThresholdEB &&
           pfrh_pt2[i] > seedPt2ThresholdEB) ||
          ((pfrh_layer[i] == PFLayer::ECAL_ENDCAP) && pfrh_energy[i] > seedEThresholdEE &&
           pfrh_pt2[i] > seedPt2ThresholdEE)) {
        pfrh_isSeed[i] = 1;
        for (int k = 0; k < nNeigh; k++) {
          if (neigh8_Ind[nNeigh * i + k] < 0)
            continue;
          if (pfrh_energy[i] < pfrh_energy[neigh8_Ind[nNeigh * i + k]]) {
            pfrh_isSeed[i] = 0;
            pfrh_topoId[i] = -1;
            break;
          }
        }
      } else {
        pfrh_topoId[i] = -1;
        pfrh_isSeed[i] = 0;
      }
    }
  }

  __global__ void seedingKernel_ECAL_serialize(size_t size,
                                               const float* __restrict__ pfrh_energy,
                                               const float* __restrict__ pfrh_pt2,
                                               int* pfrh_isSeed,
                                               int* pfrh_topoId,
                                               const int* __restrict__ pfrh_layer,
                                               const int* __restrict__ neigh8_Ind) {
    //int i = threadIdx.x+blockIdx.x*blockDim.x;

    for (int i = 0; i < size; i++) {
      if ((pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i] > seedEThresholdEB &&
           pfrh_pt2[i] > seedPt2ThresholdEB) ||
          ((pfrh_layer[i] == PFLayer::ECAL_ENDCAP) && pfrh_energy[i] > seedEThresholdEE &&
           pfrh_pt2[i] > seedPt2ThresholdEE)) {
        pfrh_isSeed[i] = 1;
        for (int k = 0; k < nNeigh; k++) {
          if (neigh8_Ind[nNeigh * i + k] < 0)
            continue;
          if (pfrh_energy[i] < pfrh_energy[neigh8_Ind[nNeigh * i + k]]) {
            pfrh_isSeed[i] = 0;
            pfrh_topoId[i] = -1;
            break;
          }
        }
      } else {
        pfrh_topoId[i] = -1;
        pfrh_isSeed[i] = 0;
      }
    }
  }

  __global__ void topoKernel_ECAL(size_t size,
                                  const float* __restrict__ pfrh_energy,
                                  int* pfrh_topoId,
                                  const int* __restrict__ pfrh_layer,
                                  const int* __restrict__ neigh8_Ind) {
    for (int j = 0; j < 16; j++) {
      int l = threadIdx.x + blockIdx.x * blockDim.x;
      if (l < size) {
        //printf("layer: %d",pfrh_layer[l]);
        for (int k = 0; k < nNeigh; k++) {
          if (neigh8_Ind[nNeigh * l + k] > -1 && pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNeigh * l + k]] &&
              ((pfrh_layer[l] == PFLayer::ECAL_ENDCAP && pfrh_energy[l] > topoEThresholdEE) ||
               (pfrh_layer[l] == PFLayer::ECAL_BARREL && pfrh_energy[l] > topoEThresholdEB))) {
            pfrh_topoId[l] = pfrh_topoId[neigh8_Ind[nNeigh * l + k]];
          }
        }
      }  //loop over neighbours end

    }  //loop over neumann neighbourhood clustering end
  }

  __global__ void topoKernel_ECALV2(size_t size,
                                    const float* __restrict__ pfrh_energy,
                                    int* pfrh_topoId,
                                    const int* __restrict__ pfrh_layer,
                                    const int* __restrict__ neigh8_Ind) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    int k = (threadIdx.y + blockIdx.y * blockDim.y) % nNeigh;

    //if(l<size && k<8) {
    if (l < size) {
      while (neigh8_Ind[nNeigh * l + k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNeigh * l + k]] &&
             (((pfrh_layer[l] == PFLayer::ECAL_ENDCAP && pfrh_energy[l] > topoEThresholdEE) ||
               (pfrh_layer[l] == PFLayer::ECAL_BARREL && pfrh_energy[l] > topoEThresholdEB)) &&
              ((pfrh_layer[neigh8_Ind[nNeigh * l + k]] == PFLayer::ECAL_ENDCAP &&
                pfrh_energy[neigh8_Ind[nNeigh * l + k]] > topoEThresholdEE) ||
               (pfrh_layer[neigh8_Ind[nNeigh * l + k]] == PFLayer::ECAL_BARREL &&
                pfrh_energy[neigh8_Ind[nNeigh * l + k]] > topoEThresholdEB)))) {
        if (pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNeigh * l + k]]) {
          atomicMax(&pfrh_topoId[neigh8_Ind[nNeigh * l + k]], pfrh_topoId[l]);
        }
        if (pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNeigh * l + k]]) {
          atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNeigh * l + k]]);
        }
      }
    }
  }

  __global__ void topoKernel_ECAL_serialize(size_t size,
                                            const float* __restrict__ pfrh_energy,
                                            int* pfrh_topoId,
                                            const int* __restrict__ pfrh_layer,
                                            const int* __restrict__ neigh8_Ind) {
    for (int l = 0; l < size; l++) {
      for (int k = 0; k < 8; k++) {
        while (neigh8_Ind[nNeigh * l + k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNeigh * l + k]] &&
               (((pfrh_layer[l] == PFLayer::ECAL_ENDCAP && pfrh_energy[l] > topoEThresholdEE) ||
                 (pfrh_layer[l] == PFLayer::ECAL_BARREL && pfrh_energy[l] > topoEThresholdEB)) &&
                ((pfrh_layer[neigh8_Ind[nNeigh * l + k]] == PFLayer::ECAL_ENDCAP &&
                  pfrh_energy[neigh8_Ind[nNeigh * l + k]] > topoEThresholdEE) ||
                 (pfrh_layer[neigh8_Ind[nNeigh * l + k]] == PFLayer::ECAL_BARREL &&
                  pfrh_energy[neigh8_Ind[nNeigh * l + k]] > topoEThresholdEB)))) {
          if (pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNeigh * l + k]]) {
            atomicMax(&pfrh_topoId[neigh8_Ind[nNeigh * l + k]], pfrh_topoId[l]);
          }
          if (pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNeigh * l + k]]) {
            atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNeigh * l + k]]);
          }
        }
      }
    }
  }

  __global__ void printFracs(size_t nRH,
                             float* pcrhfrac,
                             int* pcrhfracind,
                             int* topoSeedCount,
                             int* topoRHCount,
                             int* seedFracOffsets,
                             int* topoSeedOffsets,
                             int* topoSeedList) {
    for (int topoId = 0; topoId < (int)nRH; topoId++) {
      if (topoSeedCount[topoId] <= 0)
        continue;
      int nSeeds = topoSeedCount[topoId];
      int nRHTopo = topoRHCount[topoId];
      int nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      int topoSeedBegin = topoSeedOffsets[topoId];

      auto getSeedRhIdx = [&](int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
          printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds);
          return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
      };

      auto getRhFracIdx = [&](int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
      };

      auto getRhFrac = [&](int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
      };

      printf("\n\nTopo cluster %d has %d seeds and %d other rechits\n", topoId, nSeeds, nRHTopo - nSeeds);
      for (int s = 0; s < nSeeds; s++) {
        int i = getSeedRhIdx(s);
        printf("  PF cluster with seed %d\n", i);
        for (int r = 0; r < nRHNotSeed; r++) {
          int fracidx = getRhFracIdx(s, r);
          float frac = getRhFrac(s, r);

          printf("\trechit %d%s\tfracidx = %d\tfrac = %f\n", r, (r == 0 ? "(seed)" : "\t"), fracidx, frac);

          //                if (r == 0)
          //                    printf("\trechit %d(seed)\tfracidx = %d\tfrac = %f\n", r, fracidx, frac);
          //                else
          //                    printf("\trechit %d\t\tfracidx = %d\tfrac = %f\n", r, fracidx, frac);
        }
      }
    }
  }

  __global__ void fastCluster(size_t nRH,
                              const float* __restrict__ pfrh_x,
                              const float* __restrict__ pfrh_y,
                              const float* __restrict__ pfrh_z,
                              const float* __restrict__ geomAxis_x,
                              const float* __restrict__ geomAxis_y,
                              const float* __restrict__ geomAxis_z,
                              const float* __restrict__ pfrh_energy,
                              int* pfrh_topoId,
                              int* pfrh_isSeed,
                              const int* __restrict__ pfrh_layer,
                              const int* __restrict__ neigh8_Ind,
                              float* pcrhfrac,
                              int* pcrhfracind,
                              float* fracSum,
                              int* rhCount,
                              int* topoSeedCount,
                              int* topoRHCount,
                              int* seedFracOffsets,
                              int* topoSeedOffsets,
                              int* topoSeedList,
                              float4* clusterPos,
                              float4* prevClusterPos,
                              float4* linearClusterPos,
                              float4* convClusterPos,
                              float* clusterEnergy,
                              float* clusterT0,
                              int* pfcIter) {
    int topoId = blockIdx.x;

    if (topoId > -1 && topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 &&
        topoRHCount[topoId] != topoSeedCount[topoId]) {
      //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
      __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
      __shared__ float tol, diff, diff2;
      __shared__ bool notDone, debug;
      if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
        gridStride = blockDim.x * gridDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        //debug = (topoId == 206 || topoId == 201 || topoId == 213 || topoId == 205 || topoId == 207 || topoId == 212 || topoId == 211 || topoId == 217 || topoId == 214) ? true : false;
      }
      __syncthreads();

      auto getSeedRhIdx = [&](int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
          printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds);
          return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
      };

      auto getRhFracIdx = [&](int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
      };

      auto getRhFrac = [&](int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
      };

      if (debug) {
        if (threadIdx.x == 0) {
          printf("\n===========================================================================================\n");
          printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
          for (int s = 0; s < nSeeds; s++) {
            if (s != 0)
              printf(", ");
            printf("%d", getSeedRhIdx(s));
          }
          if (nRHTopo == nSeeds) {
            printf(")\n\n");
          } else {
            printf(") and other rechits (");
            for (int r = 1; r < nRHNotSeed; r++) {
              if (r != 1)
                printf(", ");
              printf("%d", getRhFracIdx(0, r));
            }
            printf(")\n\n");
          }
        }
        __syncthreads();
      }

      auto computeDepthPos = [&](float4& pos4,
                                 float4& linear_pos,
                                 float _frac,
                                 int rhInd,
                                 float _clusterT0,
                                 float maxDepthFront,
                                 float totalClusterEnergy,
                                 float logETotInv,
                                 bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float4 rechitAxis = make_float4(geomAxis_x[rhInd], geomAxis_y[rhInd], geomAxis_z[rhInd], 1.0);
        float weight = 0.0;

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        if (rh_energy > 0.0)
          weight = fmaxf(0.0, conv_W0 + logf(rh_energy) + logETotInv);
        const float depth = maxDepthFront - mag(rechitPos);

        if (isDebug)
          printf("\t\t\trechit %d: w=%f\tfrac=%f\tdepth=%f\trh_energy=%f\trhPos=(%f, %f, %f)\tdeltaPos=(%f, %f, %f)\n",
                 rhInd,
                 weight,
                 _frac,
                 depth,
                 rh_energy,
                 rechitPos.x,
                 rechitPos.y,
                 rechitPos.z,
                 weight * (rechitPos.x + depth * geomAxis_x[rhInd]),
                 weight * (rechitPos.y + depth * geomAxis_y[rhInd]),
                 weight * (rechitPos.z + depth * geomAxis_z[rhInd]));

        atomicAdd(&pos4.x, weight * (rechitPos.x + depth * geomAxis_x[rhInd]));
        atomicAdd(&pos4.y, weight * (rechitPos.y + depth * geomAxis_y[rhInd]));
        atomicAdd(&pos4.z, weight * (rechitPos.z + depth * geomAxis_z[rhInd]));
        atomicAdd(&pos4.w, weight);  //  position_norm

        if (pos4.w > 0)
          return;  // No need to compute position with linear weights if position_norm > 0
        // Compute linear weights
        float lin_weight = 0.0;
        if (rh_energy > 0.0)
          lin_weight = rh_energy / totalClusterEnergy;

        atomicAdd(&linear_pos.x, lin_weight * rechitPos.x);
        atomicAdd(&linear_pos.y, lin_weight * rechitPos.y);
        atomicAdd(&linear_pos.z, lin_weight * rechitPos.z);
        atomicAdd(&linear_pos.w, lin_weight);
      };

      auto computeGenericPos = [&](float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = logWeightDenominatorInv;

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm = (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
        if (isDebug)
          printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
                 rhInd,
                 norm,
                 _frac,
                 rh_energy,
                 rechitPos.x,
                 rechitPos.y,
                 rechitPos.z);

        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);  //  position_norm
      };

      // Set initial cluster position (energy) to seed rechit position (energy)
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[i] = make_float4(0., 0., 0., 0.);
        linearClusterPos[i] = make_float4(0., 0., 0., 0.);
        convClusterPos[i] = make_float4(0., 0., 0., 0.);
        clusterEnergy[i] = pfrh_energy[i];

        clusterT0[i] = 0.;
        if (pfrh_layer[i] == PFLayer::ECAL_BARREL)
          clusterT0[i] = conv_T0_EB;
        else if (pfrh_layer[i] == PFLayer::ECAL_ENDCAP)
          clusterT0[i] = conv_T0_EE;

        float seedEta = etaFromCartesian(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
        float absSeedEta = fabsf(seedEta);
        if (absSeedEta > preshowerStartEta && absSeedEta < preshowerEndEta) {
          if (seedEta > 0) {
            clusterT0[i] = conv_T0_ES;
          } else if (seedEta < 0) {
            clusterT0[i] = conv_T0_ES;
          } else {
            printf("SOMETHING WRONG WITH THIS CLUSTER ETA!\n");
          }
        }

        float logETot_inv = -logf(pfrh_energy[i]);
        float maxDepth = conv_X0 * (clusterT0[i] - logETot_inv);
        float maxToFront = mag(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
        float maxDepthPlusFront = maxDepth + maxToFront;
        computeDepthPos(prevClusterPos[i],
                        linearClusterPos[i],
                        1.0,
                        i,
                        clusterT0[i],
                        maxDepthPlusFront,
                        pfrh_energy[i],
                        logETot_inv,
                        debug);
        prevClusterPos[i].x /= prevClusterPos[i].w;
        prevClusterPos[i].y /= prevClusterPos[i].w;
        prevClusterPos[i].z /= prevClusterPos[i].w;
      }
      __syncthreads();

      while (notDone) {
        if (debug && threadIdx.x == 0) {
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }

        // Reset fracSum, rhCount, pcrhfrac
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {
          int j = getRhFracIdx(0, r);
          fracSum[j] = 0.;
          rhCount[j] = 1;

          for (int s = 0; s < nSeeds; s++) {
            int i = getSeedRhIdx(s);
            pcrhfrac[seedFracOffsets[i] + r] = -1.;
          }
        }
        __syncthreads();

        // First calculate rechit fraction sum
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                              // PF clusters
            int i = getSeedRhIdx(s);
            int j = getRhFracIdx(s, r);

            if (debug && threadIdx.x == 0) {
              printf(
                  "\tCluster %d (seed %d) using prev convergence pos = (%f, %f, %f) and cluster position = (%f, %f, "
                  "%f)\n",
                  s,
                  i,
                  prevClusterPos[i].x,
                  prevClusterPos[i].y,
                  prevClusterPos[i].z,
                  clusterPos[i].x,
                  clusterPos[i].y,
                  clusterPos[i].z);
            }

            float dist2 = (clusterPos[i].x - pfrh_x[j]) * (clusterPos[i].x - pfrh_x[j]) +
                          (clusterPos[i].y - pfrh_y[j]) * (clusterPos[i].y - pfrh_y[j]) +
                          (clusterPos[i].z - pfrh_z[j]) * (clusterPos[i].z - pfrh_z[j]);
            float d2 = dist2 / showerSigma2;
            float fraction = -1.;

            if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
              fraction = clusterEnergy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
            } else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
              fraction = clusterEnergy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
            }
            if (fraction == -1.)
              printf("FRACTION is NEGATIVE!!!");

            if (pfrh_isSeed[j] != 1) {
              atomicAdd(&fracSum[j], fraction);
            }
          }
        }
        __syncthreads();

        // Calculate rechit fractions
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                              // PF clusters
            int i = getSeedRhIdx(s);
            int j = getRhFracIdx(s, r);

            if (pfrh_isSeed[j] != 1) {
              float dist2 = (clusterPos[i].x - pfrh_x[j]) * (clusterPos[i].x - pfrh_x[j]) +
                            (clusterPos[i].y - pfrh_y[j]) * (clusterPos[i].y - pfrh_y[j]) +
                            (clusterPos[i].z - pfrh_z[j]) * (clusterPos[i].z - pfrh_z[j]);

              float d2 = dist2 / showerSigma2;
              float fraction = -1.;

              if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
                fraction = clusterEnergy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
              } else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
                fraction = clusterEnergy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
              }
              if (fraction == -1.)
                printf("FRACTION is NEGATIVE!!!");

              if (fracSum[j] > minFracTot) {
                float fracpct = fraction / fracSum[j];
                if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                  pcrhfrac[seedFracOffsets[i] + r] = fracpct;
                } else {
                  if (debug)
                    printf("\trechit %d fracSum = %f\tfracpct = %f\td2 = %f\tminFracToKeep = %f\n",
                           j,
                           fracSum[j],
                           fracpct,
                           d2,
                           minFracToKeep);
                  pcrhfrac[seedFracOffsets[i] + r] = -1;
                }
              } else {
                if (debug)
                  printf("\trechit %d fracSum = %f LESS than minFracTot = %f\n", j, fracSum[j], minFracTot);
                pcrhfrac[seedFracOffsets[i] + r] = -1;
              }
            }
          }
        }
        __syncthreads();

        if (debug && threadIdx.x == 0)
          printf("Computing cluster position for topoId %d\n", topoId);

        // Reset cluster energy and positions
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
          int i = getSeedRhIdx(s);
          clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
          linearClusterPos[i] = make_float4(0., 0., 0., 0.);
          convClusterPos[i] = make_float4(0., 0., 0., 0.);
          clusterEnergy[i] = 0;
        }
        __syncthreads();

        // Recalculate cluster position and energy
        for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                          // PF clusters
            int i = getSeedRhIdx(s);                                  // Seed index

            // Calculate cluster energy by summing rechit fractional energies
            int j = getRhFracIdx(s, r);
            float frac = getRhFrac(s, r);

            if (frac > -0.5) {
              //if (debug)
              //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
              atomicAdd(&clusterEnergy[i], frac * pfrh_energy[j]);

              bool updateClusterPos = false;
              if (nSeeds == 1) {
                if (debug && threadIdx.x == 0)
                  printf("\t\tThis topo cluster has a single seed.\n");
                //computeClusterPos(clusterPos[i], frac, j, debug);
                updateClusterPos = true;
              } else {
                if (j == i) {
                  // This is the seed
                  updateClusterPos = true;
                } else {
                  // Check if this is one of the neighboring rechits
                  for (int k = 0; k < nNeigh; k++) {
                    if (neigh8_Ind[nNeigh * i + k] < 0)
                      continue;
                    if (neigh8_Ind[nNeigh * i + k] == j) {
                      // Found it
                      if (debug)
                        printf("\t\tRechit %d is one of the 8 neighbors of seed %d\n", j, i);
                      updateClusterPos = true;
                      break;
                    }
                  }
                }
              }
              if (updateClusterPos)
                computeGenericPos(clusterPos[i], frac, j, debug);
            }
          }
        }
        __syncthreads();

        // ECAL 2D depth cluster position calculation
        for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                          // PF clusters
            int i = getSeedRhIdx(s);                                  // Seed index
            int j = getRhFracIdx(s, r);
            float frac = getRhFrac(s, r);

            if (frac > -0.5) {
              float logETot_inv = -logf(clusterEnergy[i]);
              float maxDepth = conv_X0 * (clusterT0[i] - logETot_inv);
              float maxToFront = mag(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
              float maxDepthPlusFront = maxDepth + maxToFront;
              computeDepthPos(convClusterPos[i],
                              linearClusterPos[i],
                              frac,
                              j,
                              clusterT0[i],
                              maxDepthPlusFront,
                              clusterEnergy[i],
                              logETot_inv,
                              debug);
            }
            //else if (debug)
            //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
          }
        }
        __syncthreads();

        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
          int i = getSeedRhIdx(s);  // Seed index
          // Generic position calculation
          if (clusterPos[i].w >= minAllowedNormalization) {
            // Divide by position norm
            clusterPos[i].x /= clusterPos[i].w;
            clusterPos[i].y /= clusterPos[i].w;
            clusterPos[i].z /= clusterPos[i].w;
            if (debug)
              printf("\tCluster %d (seed %d) energy = %f\tgeneric pos = (%f, %f, %f)\n",
                     s,
                     i,
                     clusterEnergy[i],
                     clusterPos[i].x,
                     clusterPos[i].y,
                     clusterPos[i].z);
          } else {
            if (debug)
              printf("\tGeneric pos calc: Cluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                     s,
                     i,
                     clusterPos[i].w,
                     minAllowedNormalization);
            clusterPos[i].x = 0.0;
            clusterPos[i].y = 0.0;
            clusterPos[i].z = 0.0;
          }

          // ECAL depth corrected position
          if (convClusterPos[i].w >= conv_minAllowedNormalization && convClusterPos[i].w >= 0.0) {
            // Divide by position norm
            convClusterPos[i].x /= convClusterPos[i].w;
            convClusterPos[i].y /= convClusterPos[i].w;
            convClusterPos[i].z /= convClusterPos[i].w;

            if (debug)
              printf("\tCluster %d (seed %d) energy = %f\t2D depth cor pos = (%f, %f, %f)\n",
                     s,
                     i,
                     clusterEnergy[i],
                     convClusterPos[i].x,
                     convClusterPos[i].y,
                     convClusterPos[i].z);
          } else if (fabsf(convClusterPos[i].w) < 1e-5 && linearClusterPos[i].w >= conv_minAllowedNormalization) {
            convClusterPos[i].x = linearClusterPos[i].x / linearClusterPos[i].w;
            convClusterPos[i].y = linearClusterPos[i].y / linearClusterPos[i].w;
            convClusterPos[i].z = linearClusterPos[i].z / linearClusterPos[i].w;
            if (debug)
              printf("\tCluster %d (seed %d) falling back to linear weights!\tenergy = %f\tpos = (%f, %f, %f)\n",
                     s,
                     i,
                     clusterEnergy[i],
                     convClusterPos[i].x,
                     convClusterPos[i].y,
                     convClusterPos[i].z);
          } else {
            if (debug)
              printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                     s,
                     i,
                     linearClusterPos[i].w,
                     conv_minAllowedNormalization);
            convClusterPos[i].x = 0.0;
            convClusterPos[i].y = 0.0;
            convClusterPos[i].z = 0.0;
            //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
          }
        }
        if (threadIdx.x == 0)
          diff2 = -1.;
        __syncthreads();

        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
          int i = getSeedRhIdx(s);
          float delta2 = dR2(prevClusterPos[i], convClusterPos[i]);
          if (debug)
            printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
          if (delta2 > diff2) {
            diff2 = delta2;
            if (debug)
              printf("\t\tNew diff2 = %f\n", diff2);
          }

          // Set previous cluster position to convergence pos
          prevClusterPos[i] = convClusterPos[i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
          diff = sqrtf(diff2);
          iter++;
          notDone = (diff > tol) && (iter < maxIterations);
          if (debug) {
            if (diff > tol)
              printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
            else if (debug)
              printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
          }
        }
        __syncthreads();
      }  // end while iter loop
      if (threadIdx.x == 0)
        pfcIter[topoId] = iter;
    } else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 ||
                                    (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
      // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
      pfcIter[topoId] = 0;
    }
  }

  __global__ void fastCluster_serialize(size_t nRH,
                                        const float* __restrict__ pfrh_x,
                                        const float* __restrict__ pfrh_y,
                                        const float* __restrict__ pfrh_z,
                                        const float* __restrict__ geomAxis_x,
                                        const float* __restrict__ geomAxis_y,
                                        const float* __restrict__ geomAxis_z,
                                        const float* __restrict__ pfrh_energy,
                                        int* pfrh_topoId,
                                        int* pfrh_isSeed,
                                        const int* __restrict__ pfrh_layer,
                                        const int* __restrict__ neigh8_Ind,
                                        float* pcrhfrac,
                                        int* pcrhfracind,
                                        float* fracSum,
                                        int* rhCount) {
    for (int topoId = 0; topoId < (int)nRH; topoId++) {
      int iter = 0;
      int nSeeds = 0;
      int nRHTopo = 0;
      if (topoId > -1 && topoId < nRH) {
        //int seeds[25];
        int seeds[75];
        int rechits[150];
        // First determine how many rechits are in this topo cluster
        for (int r = 0; r < nRH; r++) {
          if (pfrh_topoId[r] == topoId) {
            // Found a rechit belonging to this topo cluster
            rechits[nRHTopo] = r;
            nRHTopo++;
            if (pfrh_isSeed[r]) {
              // This rechit is a seed
              seeds[nSeeds] = r;
              nSeeds++;
            }
          }
        }
        if (nSeeds == 0)
          continue;  // No seeds found for this topoId. Skip it

        //bool debug = true;
        bool debug = false;

        if (debug) {
          printf("\n===========================================================================================\n");
          printf("Processing topo cluster %d with seeds (", topoId);
          for (int s = 0; s < nSeeds; s++) {
            if (s != 0)
              printf(", ");
            printf("%d", seeds[s]);
          }
          printf(") and rechits (");
          for (int r = 0; r < nRHTopo; r++) {
            if (r != 0)
              printf(", ");
            printf("%d", rechits[r]);
          }
          printf(")");
        }

        //float tolScaling2 = std::pow(std::max(1.0, nSeeds - 1.0), 4.0);     // Tolerance scaling squared
        float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // Tolerance scaling

        float4 prevClusterPos[75], linearClusterPos[75], clusterPos[75],
            convClusterPos[75];  //  W component is position norm
        float clusterEnergy[75];
        //float prevClusterEnergy[75];

        auto computeDepthPosFromArrays = [&](float4& pos4,
                                             float4& linear_pos,
                                             float _frac,
                                             int rhInd,
                                             float _clusterT0,
                                             float maxDepthFront,
                                             float totalClusterEnergy,
                                             float logETotInv,
                                             bool isDebug) {
          float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
          float4 rechitAxis = make_float4(geomAxis_x[rhInd], geomAxis_y[rhInd], geomAxis_z[rhInd], 1.0);
          float weight = 0.0;

          const auto rh_energy = pfrh_energy[rhInd] * _frac;
          if (rh_energy > 0.0)
            weight = fmaxf(0.0, conv_W0 + logf(rh_energy) + logETotInv);
          const float depth = maxDepthFront - mag(rechitPos);

          if (isDebug)
            printf(
                "\t\t\trechit %d: w=%f\tfrac=%f\tdepth=%f\trh_energy=%f\trhPos=(%f, %f, %f)\tdeltaPos=(%f, %f, %f)\n",
                rhInd,
                weight,
                _frac,
                depth,
                rh_energy,
                rechitPos.x,
                rechitPos.y,
                rechitPos.z,
                weight * (rechitPos.x + depth * geomAxis_x[rhInd]),
                weight * (rechitPos.y + depth * geomAxis_y[rhInd]),
                weight * (rechitPos.z + depth * geomAxis_z[rhInd]));

          pos4.x += weight * (rechitPos.x + depth * geomAxis_x[rhInd]);
          pos4.y += weight * (rechitPos.y + depth * geomAxis_y[rhInd]);
          pos4.z += weight * (rechitPos.z + depth * geomAxis_z[rhInd]);
          pos4.w += weight;  //  position_norm

          if (pos4.w > 0)
            return;  // No need to compute position with linear weights if position_norm > 0
          // Compute linear weights
          float lin_weight = 0.0;
          if (rh_energy > 0.0)
            lin_weight = rh_energy / totalClusterEnergy;

          linear_pos.x += lin_weight * rechitPos.x;
          linear_pos.y += lin_weight * rechitPos.y;
          linear_pos.z += lin_weight * rechitPos.z;
          linear_pos.w += lin_weight;
        };

        auto computeFromArrays = [&](float4& pos4, float _frac, int rhInd, bool isDebug) {
          float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
          float threshold = logWeightDenominatorInv;
          //            float threshold = 0.0;
          //            if(pfrh_layer[rhInd] == PFLayer::ECAL_BARREL) {
          //                threshold = 1. / recHitEnergyNormInvEB; // This number needs to be inverted
          //            }
          //            else if (pfrh_layer[rhInd] == PFLayer::ECAL_ENDCAP) { threshold = 1. / recHitEnergyNormInvEE; }

          const auto rh_energy = pfrh_energy[rhInd] * _frac;
          const auto norm = (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
          if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
                   rhInd,
                   norm,
                   _frac,
                   rh_energy,
                   rechitPos.x,
                   rechitPos.y,
                   rechitPos.z);

          pos4.x += rechitPos.x * norm;
          pos4.y += rechitPos.y * norm;
          pos4.z += rechitPos.z * norm;
          pos4.w += norm;  //  position_norm
        };
        /*
        auto compute = [&] (float4& pos4, float& clusterEn, int seedInd, int rhInd) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float threshold = 0.0;
            // Search for this rechit index in fraction arrays
            float _frac = -1.0;
            for (int _n = seedInd*maxSize; _n < ((seedInd+1)*maxSize); _n++) {
                if (pcrhfracind[_n] == rhInd) {
                    // Found it
                    _frac = pcrhfrac[_n];
                    break;
                }
            }
            if (_frac < 0)
                printf("Warning: negative rechitfrac found for seed %d rechit %d!\n", seedInd, rhInd);
            if(pfrh_layer[rhInd] == 1) {
                threshold = 1. / recHitEnergyNormInvEB; // This number needs to be inverted
            }
            else if (pfrh_layer[rhInd] == 3) { threshold = 1. / recHitEnergyNormInvEE; }

            const auto rh_energy = pfrh_energy[rhInd] * _frac;
            const auto norm =
                (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
            pos4.x += rechitPos.x * norm;
            pos4.y += rechitPos.y * norm;
            pos4.z += rechitPos.z * norm;
            pos4.w += norm;     //  position_norm

            clusterEn += rh_energy;
        };
        */
        float diff = -1.0;
        while (iter < maxIterations) {
          if (debug) {
            printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
          }
          // Reset fracSum and rhCount
          for (int r = 0; r < (int)nRH; r++) {
            fracSum[r] = 0.0;
            rhCount[r] = 1;
          }

          for (int s = 0; s < nSeeds; s++) {  // PF clusters
            int i = seeds[s];
            if (iter == 0) {
              // Set initial cluster position to seed rechit position
              clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
              prevClusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
              //prevClusterPos[s] = clusterPos[s];

              float clusterT0 = 0.0;
              if (pfrh_layer[i] == PFLayer::ECAL_BARREL)
                clusterT0 = conv_T0_EB;
              else if (pfrh_layer[i] == PFLayer::ECAL_ENDCAP)
                clusterT0 = conv_T0_EE;

              float seedEta = etaFromCartesian(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
              float absSeedEta = fabsf(seedEta);
              if (absSeedEta > preshowerStartEta && absSeedEta < preshowerEndEta) {
                if (seedEta > 0) {
                  clusterT0 = conv_T0_ES;
                } else if (seedEta < 0) {
                  clusterT0 = conv_T0_ES;
                } else {
                  printf("SOMETHING WRONG WITH THIS CLUSTER ETA!\n");
                }
              }

              float logETot_inv = -logf(pfrh_energy[i]);
              float maxDepth = conv_X0 * (clusterT0 - logETot_inv);
              float maxToFront = mag(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
              float maxDepthPlusFront = maxDepth + maxToFront;
              computeDepthPosFromArrays(prevClusterPos[s],
                                        linearClusterPos[s],
                                        1.0,
                                        i,
                                        clusterT0,
                                        maxDepthPlusFront,
                                        pfrh_energy[i],
                                        logETot_inv,
                                        debug);
              prevClusterPos[s].x /= prevClusterPos[s].w;
              prevClusterPos[s].y /= prevClusterPos[s].w;
              prevClusterPos[s].z /= prevClusterPos[s].w;
              // Set initial cluster energy to seed energy
              clusterEnergy[s] = pfrh_energy[i];
              //prevClusterEnergy[s] = 0.0;
            } else {
              prevClusterPos[s] = convClusterPos[s];
              //prevClusterEnergy[s] = clusterEnergy[s];

              // Reset cluster indices and fractions
              for (int _n = i * maxSize; _n < (i + 1) * maxSize; _n++) {
                pcrhfrac[_n] = -1.0;
                pcrhfracind[_n] = -1.0;
              }
            }
            if (debug) {
              printf(
                  "\tCluster %d (seed %d) using prev convergence pos = (%f, %f, %f) and cluster position = (%f, %f, "
                  "%f)\n",
                  s,
                  i,
                  prevClusterPos[s].x,
                  prevClusterPos[s].y,
                  prevClusterPos[s].z,
                  clusterPos[s].x,
                  clusterPos[s].y,
                  clusterPos[s].z);
            }

            for (int r = 0; r < nRHTopo; r++) {
              int j = rechits[r];
              if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
                float dist2 = (clusterPos[s].x - pfrh_x[j]) * (clusterPos[s].x - pfrh_x[j]) +
                              (clusterPos[s].y - pfrh_y[j]) * (clusterPos[s].y - pfrh_y[j]) +
                              (clusterPos[s].z - pfrh_z[j]) * (clusterPos[s].z - pfrh_z[j]);
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;

                if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
                  fraction = clusterEnergy[s] * recHitEnergyNormInvEB * expf(-0.5 * d2);
                } else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
                  fraction = clusterEnergy[s] * recHitEnergyNormInvEE * expf(-0.5 * d2);
                }
                if (fraction == -1.)
                  printf("FRACTION is NEGATIVE!!!");

                if (pfrh_isSeed[j] != 1) {
                  atomicAdd(&fracSum[j], fraction);
                }
              }
            }
          }
          for (int s = 0; s < nSeeds; s++) {  // PF clusters
            int i = seeds[s];
            for (int r = 0; r < nRHTopo; r++) {
              int j = rechits[r];
              if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
                if (i == j) {
                  pcrhfrac[i * maxSize] = 1.;
                  pcrhfracind[i * maxSize] = j;
                }
                if (pfrh_isSeed[j] != 1) {
                  float dist2 = (clusterPos[s].x - pfrh_x[j]) * (clusterPos[s].x - pfrh_x[j]) +
                                (clusterPos[s].y - pfrh_y[j]) * (clusterPos[s].y - pfrh_y[j]) +
                                (clusterPos[s].z - pfrh_z[j]) * (clusterPos[s].z - pfrh_z[j]);
                  float d2 = dist2 / showerSigma2;
                  float fraction = -1.;

                  if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
                    fraction = clusterEnergy[s] * recHitEnergyNormInvEB * expf(-0.5 * d2);
                  } else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
                    fraction = clusterEnergy[s] * recHitEnergyNormInvEE * expf(-0.5 * d2);
                  }
                  if (fraction == -1.)
                    printf("FRACTION is NEGATIVE!!!");

                  if (fracSum[j] > minFracTot) {
                    float fracpct = fraction / fracSum[j];
                    if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                      int k = atomicAdd(&rhCount[i], 1);
                      pcrhfrac[i * maxSize + k] = fracpct;
                      pcrhfracind[i * maxSize + k] = j;
                    }
                  }
                }
              }
            }
          }

          if (debug)
            printf("Computing cluster position for topoId %d\n", topoId);
          // Recalculate position
          for (int s = 0; s < nSeeds; s++) {  // PF clusters
            int i = seeds[s];

            if (debug) {
              printf("\tNow on seed %d\t\tneigh8Ind = [", i);
              for (int k = 0; k < nNeigh; k++) {
                if (k != 0)
                  printf(", ");
                printf("%d", neigh8_Ind[nNeigh * i + k]);
              }
              printf("]\n");
            }
            // Zero out cluster position and energy
            clusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
            linearClusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
            convClusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[s] = 0;
            float clusterT0 = 0.0, maxDepth = 0.0, maxToFront = 0.0, maxDepthPlusFront = 0.0, logETot_inv = 0.0;
            if (pfrh_layer[i] == PFLayer::ECAL_BARREL)
              clusterT0 = conv_T0_EB;
            else if (pfrh_layer[i] == PFLayer::ECAL_ENDCAP)
              clusterT0 = conv_T0_EE;

            float seedEta = etaFromCartesian(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
            float absSeedEta = fabsf(seedEta);
            if (absSeedEta > preshowerStartEta && absSeedEta < preshowerEndEta) {
              if (seedEta > 0) {
                clusterT0 = conv_T0_ES;
                if (debug)
                  printf("\t\t## This cluster is in esPlus! ##\n");
              } else if (seedEta < 0) {
                clusterT0 = conv_T0_ES;
                if (debug)
                  printf("\t\t## This cluster is in esMinus! ##\n");
              } else {
                printf("SOMETHING WRONG WITH THIS CLUSTER ETA!\n");
              }
            }

            // Calculate cluster energy by summing rechit fractional energies
            for (int r = 0; r < nRHTopo; r++) {
              int j = rechits[r];
              float frac = -1.0;
              int _n = -1;
              if (j == i) {
                // This is the seed
                frac = 1.0;
                _n = i * maxSize;
              } else {
                for (_n = i * maxSize; _n < ((i + 1) * maxSize); _n++) {
                  if (pcrhfracind[_n] == j) {
                    // Found it
                    frac = pcrhfrac[_n];
                    break;
                  }
                }
              }
              if (frac > -0.5) {
                //if (debug)
                //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                clusterEnergy[s] += frac * pfrh_energy[j];

                // Do generic cluster position calculation
                if (nSeeds == 1) {
                  if (debug)
                    printf("\t\tThis topo cluster has a single seed.\n");
                  computeFromArrays(clusterPos[s], frac, j, debug);
                } else {
                  if (j == i) {
                    // This is the seed
                    computeFromArrays(clusterPos[s], frac, j, debug);
                  } else {
                    // Check if this is one of the neighboring rechits
                    for (int k = 0; k < nNeigh; k++) {
                      if (neigh8_Ind[nNeigh * i + k] < 0)
                        continue;
                      if (neigh8_Ind[nNeigh * i + k] == j) {
                        // Found it
                        if (debug)
                          printf("\t\tRechit %d is one of the 8 neighbors of seed %d\n", j, i);
                        computeFromArrays(clusterPos[s], frac, j, debug);
                      }
                    }
                  }
                }
              }
            }

            logETot_inv = -logf(clusterEnergy[s]);
            maxDepth = conv_X0 * (clusterT0 - logETot_inv);
            maxToFront = mag(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
            maxDepthPlusFront = maxDepth + maxToFront;

            // ECAL 2D depth cluster position calculation
            for (int r = 0; r < nRHTopo; r++) {
              int j = rechits[r];
              float frac = -1.0;
              int _n = -1;
              if (j == i) {
                // This is the seed
                frac = 1.0;
                _n = i * maxSize;
              } else {
                for (_n = i * maxSize; _n < ((i + 1) * maxSize); _n++) {
                  if (pcrhfracind[_n] == j) {
                    // Found it
                    frac = pcrhfrac[_n];
                    break;
                  }
                }
              }
              if (frac > -0.5) {
                computeDepthPosFromArrays(convClusterPos[s],
                                          linearClusterPos[s],
                                          frac,
                                          j,
                                          clusterT0,
                                          maxDepthPlusFront,
                                          clusterEnergy[s],
                                          logETot_inv,
                                          debug);
              }
              //else if (debug)
              //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
            }

            // Generic position calculation
            if (clusterPos[s].w >= minAllowedNormalization) {
              // Divide by position norm
              clusterPos[s].x /= clusterPos[s].w;
              clusterPos[s].y /= clusterPos[s].w;
              clusterPos[s].z /= clusterPos[s].w;
              if (debug)
                printf("\tCluster %d (seed %d) energy = %f\tgeneric pos = (%f, %f, %f)\n",
                       s,
                       i,
                       clusterEnergy[s],
                       clusterPos[s].x,
                       clusterPos[s].y,
                       clusterPos[s].z);
            } else {
              if (debug)
                printf("\tGeneric pos calc: Cluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                       s,
                       i,
                       clusterPos[s].w,
                       minAllowedNormalization);
              clusterPos[s].x = 0.0;
              clusterPos[s].y = 0.0;
              clusterPos[s].z = 0.0;
            }

            // ECAL depth corrected position
            if (convClusterPos[s].w >= conv_minAllowedNormalization && convClusterPos[s].w >= 0.0) {
              // Divide by position norm
              convClusterPos[s].x /= convClusterPos[s].w;
              convClusterPos[s].y /= convClusterPos[s].w;
              convClusterPos[s].z /= convClusterPos[s].w;

              if (debug)
                printf("\tCluster %d (seed %d) energy = %f\t2D depth cor pos = (%f, %f, %f)\n",
                       s,
                       i,
                       clusterEnergy[s],
                       convClusterPos[s].x,
                       convClusterPos[s].y,
                       convClusterPos[s].z);
            } else if (fabsf(convClusterPos[s].w) < 1e-5 && linearClusterPos[s].w >= conv_minAllowedNormalization) {
              convClusterPos[s].x = linearClusterPos[s].x / linearClusterPos[s].w;
              convClusterPos[s].y = linearClusterPos[s].y / linearClusterPos[s].w;
              convClusterPos[s].z = linearClusterPos[s].z / linearClusterPos[s].w;
              if (debug)
                printf("\tCluster %d (seed %d) falling back to linear weights!\tenergy = %f\tpos = (%f, %f, %f)\n",
                       s,
                       i,
                       clusterEnergy[s],
                       convClusterPos[s].x,
                       convClusterPos[s].y,
                       convClusterPos[s].z);
            } else {
              if (debug)
                printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                       s,
                       i,
                       linearClusterPos[s].w,
                       conv_minAllowedNormalization);
              convClusterPos[s].x = 0.0;
              convClusterPos[s].y = 0.0;
              convClusterPos[s].z = 0.0;
              //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
            }
          }

          float diff2 = 0.0;
          for (int s = 0; s < nSeeds; s++) {
            //float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
            float delta2 = dR2(prevClusterPos[s], convClusterPos[s]);
            if (debug)
              printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
            if (delta2 > diff2) {
              diff2 = delta2;
              if (debug)
                printf("\t\tNew diff2 = %f\n", diff2);
            }
          }
          //float diff = sqrtf(diff2);
          diff = sqrtf(diff2);
          iter++;
          //if (iter >= maxIterations || diff2 <= stoppingTolerance2 * tolScaling2) break;
          if (diff <= stoppingTolerance * tolScaling) {
            if (debug)
              printf("\tTopoId %d has diff = %f LESS than tolerance (terminating!)\n", topoId, diff);
            break;
          } else if (debug) {
            printf("\tTopoId %d has diff = %f greater than tolerance (continuing)\n", topoId, diff);
          }
        }
      }
    }
  }

  __global__ void fastCluster_original(size_t nRH,
                                       const float* __restrict__ pfrh_x,
                                       const float* __restrict__ pfrh_y,
                                       const float* __restrict__ pfrh_z,
                                       const float* __restrict__ geomAxis_x,
                                       const float* __restrict__ geomAxis_y,
                                       const float* __restrict__ geomAxis_z,
                                       const float* __restrict__ pfrh_energy,
                                       int* pfrh_topoId,
                                       int* pfrh_isSeed,
                                       const int* __restrict__ pfrh_layer,
                                       float* pcrhfrac,
                                       int* pcrhfracind,
                                       float* fracSum,
                                       int* rhCount) {
    int topoId = threadIdx.x + blockIdx.x * blockDim.x;  // TopoId
    int iter = 0;
    float tol = 0.0;
    int nSeeds = 0;
    int nRHTopo = 0;
    if (topoId > -1 && topoId < nRH) {
      int seeds[25];
      int rechits[60];
      // First determine how many rechits are in this topo cluster
      for (int r = 0; r < nRH; r++) {
        if (pfrh_topoId[r] == topoId) {
          // Found a rechit belonging to this topo cluster
          rechits[nRHTopo] = r;
          nRHTopo++;
          if (pfrh_isSeed[r]) {
            // This rechit is a seed
            seeds[nSeeds] = r;
            nSeeds++;
          }
        }
      }

      while (iter < 1) {
        for (int s = 0; s < nSeeds; s++) {  // PF clusters
          int i = seeds[s];
          for (int r = 0; r < nRHTopo; r++) {
            int j = rechits[r];
            if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
              float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                            (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                            (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

              float d2 = dist2 / showerSigma2;
              float fraction = -1.;

              if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
                fraction = pfrh_energy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
              } else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
                fraction = pfrh_energy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
              }
              if (fraction == -1.)
                printf("FRACTION is NEGATIVE!!!");

              if (pfrh_isSeed[j] != 1) {
                atomicAdd(&fracSum[j], fraction);
              }
            }
          }
        }
        for (int s = 0; s < nSeeds; s++) {  // PF clusters
          int i = seeds[s];
          for (int r = 0; r < nRHTopo; r++) {
            int j = rechits[r];
            if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
              if (i == j) {
                pcrhfrac[i * maxSize] = 1.;
                pcrhfracind[i * maxSize] = j;
              }
              if (pfrh_isSeed[j] != 1) {
                float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                              (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                              (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

                float d2 = dist2 / showerSigma2;
                float fraction = -1.;

                if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
                  fraction = pfrh_energy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
                }
                if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
                  fraction = pfrh_energy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
                }
                if (fraction == -1.)
                  printf("FRACTION is NEGATIVE!!!");

                if (fracSum[j] > minFracTot) {
                  float fracpct = fraction / fracSum[j];
                  if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                    int k = atomicAdd(&rhCount[i], 1);
                    pcrhfrac[i * maxSize + k] = fracpct;
                    pcrhfracind[i * maxSize + k] = j;
                  }
                }
              }
            }
          }
        }
        iter++;
        if (abs(tol) < stoppingTolerance)
          break;
      }
    }
  }

  __global__ void fastCluster_step1(size_t size,
                                    const float* __restrict__ pfrh_x,
                                    const float* __restrict__ pfrh_y,
                                    const float* __restrict__ pfrh_z,
                                    const float* __restrict__ pfrh_energy,
                                    int* pfrh_topoId,
                                    int* pfrh_isSeed,
                                    const int* __restrict__ pfrh_layer,
                                    float* pcrhfrac,
                                    int* pcrhfracind,
                                    float* fracSum,
                                    int* rhCount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if (i < size && j < size) {
      if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
        float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                      (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                      (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

        float d2 = dist2 / showerSigma2;
        float fraction = -1.;

        if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
          fraction = pfrh_energy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
        } else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
          fraction = pfrh_energy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
        }
        if (fraction == -1.)
          printf("FRACTION is NEGATIVE!!!");

        if (pfrh_isSeed[j] != 1) {
          atomicAdd(&fracSum[j], fraction);
        }
      }
    }
  }

  __global__ void fastCluster_step2(size_t size,
                                    const float* __restrict__ pfrh_x,
                                    const float* __restrict__ pfrh_y,
                                    const float* __restrict__ pfrh_z,
                                    const float* __restrict__ pfrh_energy,
                                    int* pfrh_topoId,
                                    int* pfrh_isSeed,
                                    const int* __restrict__ pfrh_layer,
                                    float* pcrhfrac,
                                    int* pcrhfracind,
                                    float* fracSum,
                                    int* rhCount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if (i < size && j < size) {
      if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
        if (i == j) {
          pcrhfrac[i * maxSize] = 1.;
          pcrhfracind[i * maxSize] = j;
        }
        if (pfrh_isSeed[j] != 1) {
          float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                        (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                        (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

          float d2 = dist2 / showerSigma2;
          float fraction = -1.;

          if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
            fraction = pfrh_energy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
          }
          if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
            fraction = pfrh_energy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
          }
          if (fraction == -1.)
            printf("FRACTION is NEGATIVE!!!");

          if (fracSum[j] > minFracTot) {
            float fracpct = fraction / fracSum[j];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
              int k = atomicAdd(&rhCount[i], 1);
              pcrhfrac[i * maxSize + k] = fracpct;
              pcrhfracind[i * maxSize + k] = j;
            }
          }

          /*
        if(d2 < 100. )
          { 
            if ((fraction/fracSum[j])>minFracToKeep){
              int k = atomicAdd(&rhCount[i],1);
              pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
              pcrhfracind[i*maxSize+k] = j;
            }
          }
        */
        }
      }
    }
  }

  __global__ void fastCluster_step1_serialize(size_t size,
                                              const float* __restrict__ pfrh_x,
                                              const float* __restrict__ pfrh_y,
                                              const float* __restrict__ pfrh_z,
                                              const float* __restrict__ pfrh_energy,
                                              int* pfrh_topoId,
                                              int* pfrh_isSeed,
                                              const int* __restrict__ pfrh_layer,
                                              float* pcrhfrac,
                                              int* pcrhfracind,
                                              float* fracSum,
                                              int* rhCount) {
    //int i = threadIdx.x+blockIdx.x*blockDim.x;
    //int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (i < size && j < size) {
          if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
            float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                          (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                          (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

            float d2 = dist2 / showerSigma2;
            float fraction = -1.;

            if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
              fraction = pfrh_energy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
            }
            if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
              fraction = pfrh_energy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
            }
            if (fraction == -1.)
              printf("FRACTION is NEGATIVE!!!");

            if (pfrh_isSeed[j] != 1 && d2 < 100) {
              atomicAdd(&fracSum[j], fraction);
            }
          }
        }
      }
    }
  }

  __global__ void fastCluster_step2_serialize(size_t size,
                                              const float* __restrict__ pfrh_x,
                                              const float* __restrict__ pfrh_y,
                                              const float* __restrict__ pfrh_z,
                                              const float* __restrict__ pfrh_energy,
                                              int* pfrh_topoId,
                                              int* pfrh_isSeed,
                                              const int* __restrict__ pfrh_layer,
                                              float* pcrhfrac,
                                              int* pcrhfracind,
                                              float* fracSum,
                                              int* rhCount) {
    //int i = threadIdx.x+blockIdx.x*blockDim.x;
    //int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (i < size && j < size) {
          if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
            if (i == j) {
              pcrhfrac[i * maxSize] = 1.;
              pcrhfracind[i * maxSize] = j;
            }
            if (pfrh_isSeed[j] != 1) {
              float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                            (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                            (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

              float d2 = dist2 / showerSigma2;
              float fraction = -1.;

              if (pfrh_layer[j] == PFLayer::ECAL_BARREL) {
                fraction = pfrh_energy[i] * recHitEnergyNormInvEB * expf(-0.5 * d2);
              }
              if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) {
                fraction = pfrh_energy[i] * recHitEnergyNormInvEE * expf(-0.5 * d2);
              }
              if (fraction == -1.)
                printf("FRACTION is NEGATIVE!!!");
              if (d2 < 100.) {
                if ((fraction / fracSum[j]) > minFracToKeep) {
                  int k = atomicAdd(&rhCount[i], 1);
                  pcrhfrac[i * maxSize + k] = fraction / fracSum[j];
                  pcrhfracind[i * maxSize + k] = j;
                }
              }
            }
          }
        }
      }
    }
  }

  // Contraction in a single block
  __global__ void topoClusterContraction(size_t size, int* pfrh_parent) {
    __shared__ int notDone;
    if (threadIdx.x == 0)
      notDone = 0;
    __syncthreads();

    do {
      volatile bool threadNotDone = false;
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        int parent = pfrh_parent[i];
        if (parent >= 0 && parent != pfrh_parent[parent]) {
          threadNotDone = true;
          pfrh_parent[i] = pfrh_parent[parent];
        }
      }
      if (threadIdx.x == 0)
        notDone = 0;
      __syncthreads();

      atomicAdd(&notDone, (int)threadNotDone);
      //if (threadNotDone) notDone = true;
      //notDone |= threadNotDone;
      __syncthreads();

    } while (notDone);
  }

  // Contraction in a single block
  __global__ void topoClusterContraction(size_t size,
                                         int* pfrh_parent,
                                         int* pfrh_isSeed,
                                         int* rhCount,
                                         int* topoSeedCount,
                                         int* topoRHCount,
                                         int* seedFracOffsets,
                                         int* topoSeedOffsets,
                                         int* topoSeedList,
                                         int* pcrhfracind,
                                         float* pcrhfrac,
                                         int* pcrhFracSize) {
    __shared__ int notDone, totalSeedOffset, totalSeedFracOffset;
    if (threadIdx.x == 0) {
      notDone = 0;
      totalSeedOffset = 0;
      totalSeedFracOffset = 0;
      *pcrhFracSize = 0;
    }
    __syncthreads();

    do {
      volatile bool threadNotDone = false;
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        int parent = pfrh_parent[i];
        if (parent >= 0 && parent != pfrh_parent[parent]) {
          threadNotDone = true;
          pfrh_parent[i] = pfrh_parent[parent];
        }
      }
      if (threadIdx.x == 0)
        notDone = 0;
      __syncthreads();

      atomicAdd(&notDone, (int)threadNotDone);
      __syncthreads();

    } while (notDone);

    // Now determine the number of seeds and rechits in each topo cluster
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
      int topoId = pfrh_parent[rhIdx];
      if (topoId > -1) {
        // Valid topo cluster
        atomicAdd(&topoRHCount[topoId], 1);
        if (pfrh_isSeed[rhIdx]) {
          atomicAdd(&topoSeedCount[topoId], 1);
        }
      }
    }
    __syncthreads();

    // Determine offsets for topo ID seed array
    for (int topoId = threadIdx.x; topoId < size; topoId += blockDim.x) {
      if (topoSeedCount[topoId] > 0) {
        // This is a valid topo ID
        int offset = atomicAdd(&totalSeedOffset, topoSeedCount[topoId]);
        topoSeedOffsets[topoId] = offset;
      }
    }
    __syncthreads();

    // Fill arrays of seed indicies per topo ID
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
      int topoId = pfrh_parent[rhIdx];
      if (topoId > -1 && pfrh_isSeed[rhIdx]) {
        // Valid topo cluster
        int k = atomicAdd(&rhCount[topoId], 1);
        topoSeedList[topoSeedOffsets[topoId] + k] = rhIdx;
      }
    }
    __syncthreads();

    // Determine seed offsets for rechit fraction array
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
      rhCount[rhIdx] = 1;  // Reset this counter array

      int topoId = pfrh_parent[rhIdx];
      if (pfrh_isSeed[rhIdx] && topoId > -1) {
        // Allot the total number of rechits for this topo cluster for rh fractions
        int offset = atomicAdd(&totalSeedFracOffset, topoRHCount[topoId]);

        // Add offset for this PF cluster seed
        seedFracOffsets[rhIdx] = offset;
        pcrhfracind[offset] = rhIdx;
        pcrhfrac[offset] = 1.;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      *pcrhFracSize = totalSeedFracOffset;
    }
  }

  __device__ __forceinline__ bool isLeftEdge(const int idx,
                                             const int nEdges,
                                             const int* __restrict__ pfrh_edgeId,
                                             const int* __restrict__ pfrh_edgeMask) {
    if (idx > 0) {
      int temp = idx - 1;
      int minVal = max(idx - 9, 0);  //  Only test up to 9 neighbors
      int tempId = 0;
      int edgeId = pfrh_edgeId[idx];
      while (temp >= minVal) {
        tempId = pfrh_edgeId[temp];
        if (edgeId != tempId) {
          // Different topo Id here!
          return true;
        } else if (pfrh_edgeMask[temp] > 0) {
          // Found adjacent edge
          return false;
        }
        temp--;
      }
    } else if (idx == 0) {
      return true;
    }

    // Invalid index
    return false;
  }

  __device__ __forceinline__ bool isRightEdge(const int idx,
                                              const int nEdges,
                                              const int* __restrict__ pfrh_edgeId,
                                              const int* __restrict__ pfrh_edgeMask) {
    // Update right
    if (idx < (nEdges - 1)) {
      int temp = idx + 1;
      int maxVal = min(idx + 9, nEdges - 1);  //  Only test up to 9 neighbors
      int tempId = 0;
      int edgeId = pfrh_edgeId[idx];
      while (temp <= maxVal) {
        tempId = pfrh_edgeId[temp];
        if (edgeId != tempId) {
          // Different topo Id here!
          return true;
        } else if (pfrh_edgeMask[temp] > 0) {
          // Found adjacent edge
          return false;
        }
        temp++;
      }
    } else if (idx == (nEdges - 1)) {
      return true;
    }

    // Overflow
    return false;
  }

  __global__ void topoClusterLinking(int nRH,
                                     int nEdges,
                                     int* pfrh_parent,
                                     int* pfrh_edgeId,
                                     int* pfrh_edgeList,
                                     int* pfrh_edgeMask,
                                     bool* pfrh_passTopoThresh,
                                     int* topoIter) {
    __shared__ bool notDone;
    __shared__ int iter, gridStride;

    int start = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
      *topoIter = 0;
      iter = 0;
      gridStride = blockDim.x * gridDim.x;  // For single block kernel this is the number of threads
    }
    __syncthreads();

    // Check if pairs in edgeId,edgeList contain a rh not passing topo threshold
    // If found, set the mask to 0
    for (int idx = start; idx < nEdges; idx += gridStride) {
      if (pfrh_passTopoThresh[pfrh_edgeId[idx]] && pfrh_passTopoThresh[pfrh_edgeList[idx]])
        pfrh_edgeMask[idx] = 1;
      else
        pfrh_edgeMask[idx] = 0;
    }

    // Begin linking loop
    do {
      if (threadIdx.x == 0) {
        notDone = false;
      }
      __syncthreads();

      // Odd linking
      for (int idx = start; idx < nEdges; idx += gridStride) {
        int i = pfrh_edgeId[idx];  // Get edge topo id
        //if (pfrh_edgeMask[idx] > 0 && pfrh_passTopoThresh[i] && isLeftEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
        if (pfrh_edgeMask[idx] > 0 && isLeftEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
          pfrh_parent[i] = (int)min(i, pfrh_edgeList[idx]);
        }
      }

      __syncthreads();

      // edgeParent
      for (int idx = start; idx < nEdges; idx += gridStride) {
        if (pfrh_edgeMask[idx] > 0) {
          int id = pfrh_edgeId[idx];          // Get edge topo id
          int neighbor = pfrh_edgeList[idx];  // Get neighbor topo id
          pfrh_edgeId[idx] = pfrh_parent[id];
          pfrh_edgeList[idx] = pfrh_parent[neighbor];

          // edgeMask set to true if elements of edgeId and edgeList are different
          if (pfrh_edgeId[idx] != pfrh_edgeList[idx]) {
            pfrh_edgeMask[idx] = 1;
            notDone = true;
          } else {
            pfrh_edgeMask[idx] = 0;
          }
        }
      }
      if (threadIdx.x == 0)
        iter++;

      __syncthreads();

      if (!notDone)
        break;

      if (threadIdx.x == 0) {
        notDone = false;
      }

      __syncthreads();

      // Even linking
      for (int idx = start; idx < nEdges; idx += gridStride) {
        int i = pfrh_edgeId[idx];  // Get edge topo id
        //if (pfrh_edgeMask[idx] > 0 && pfrh_passTopoThresh[i] && isRightEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
        if (pfrh_edgeMask[idx] > 0 && isRightEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
          pfrh_parent[i] = (int)max(i, pfrh_edgeList[idx]);
        }
      }

      __syncthreads();

      // edgeParent
      for (int idx = start; idx < nEdges; idx += gridStride) {
        if (pfrh_edgeMask[idx] > 0) {
          int id = pfrh_edgeId[idx];          // Get edge topo id
          int neighbor = pfrh_edgeList[idx];  // Get neighbor topo id
          pfrh_edgeId[idx] = pfrh_parent[id];
          pfrh_edgeList[idx] = pfrh_parent[neighbor];

          // edgeMask set to true if elements of edgeId and edgeList are different
          if (pfrh_edgeId[idx] != pfrh_edgeList[idx]) {
            pfrh_edgeMask[idx] = 1;
            notDone = true;
          } else {
            pfrh_edgeMask[idx] = 0;
          }
        }
      }
      if (threadIdx.x == 0)
        iter++;

      __syncthreads();

    } while (notDone);
    *topoIter = iter;
  }

  __global__ void fillRhfIndex(size_t nRH,
                               int* pfrh_parent,
                               int* pfrh_isSeed,
                               int* topoSeedCount,
                               int* topoRHCount,
                               int* seedFracOffsets,
                               int* rhCount,
                               int* pcrhfracind) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // i is the seed index
    int j = threadIdx.y + blockIdx.y * blockDim.y;  // j is NOT a seed

    if (i < nRH && j < nRH) {
      //if (i == debugSeedIdx) printf("This is fillRhfIndex with i = %d and j = %d\n", i, j);
      int topoId = pfrh_parent[i];
      if (topoId == pfrh_parent[j] && topoId > -1 && pfrh_isSeed[i] && !pfrh_isSeed[j]) {
        int k = atomicAdd(&rhCount[i], 1);        // Increment the number of rechit fractions for this seed
        pcrhfracind[seedFracOffsets[i] + k] = j;  // Save this rechit index
      }
    }
  }

  __global__ void resetOldFracArrays(size_t nRH, int* pcrhfracind, float* pcrhfrac) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < nRH) {
      for (int j = 0; j < 75; j++) {
        pcrhfracind[i * 75 + j] = -1;
        pcrhfrac[i * 75 + j] = -1;
      }
    }
  }

  __global__ void printPFCIter(int nRH, int* pfcIter) {
    printf("pfcIter = \n[");
    for (int i = 0; i < nRH; i++) {
      if (i != 0)
        printf(", ");
      printf("%d", pfcIter[i]);
    }
    printf("]\n\n");
  }

  void PFRechitToPFCluster_ECAL_CCLClustering(cudaStream_t cudaStream,
                                              int nRH,
                                              int nEdges,
                                              const float* __restrict__ pfrh_x,
                                              const float* __restrict__ pfrh_y,
                                              const float* __restrict__ pfrh_z,
                                              const float* __restrict__ geomAxis_x,
                                              const float* __restrict__ geomAxis_y,
                                              const float* __restrict__ geomAxis_z,
                                              const float* __restrict__ pfrh_energy,
                                              const float* __restrict__ pfrh_pt2,
                                              int* pfrh_isSeed,
                                              int* pfrh_topoId,
                                              const int* __restrict__ pfrh_layer,
                                              const int* __restrict__ neigh8_Ind,
                                              int* pfrh_edgeId,
                                              int* pfrh_edgeList,
                                              int* pfrh_edgeMask,
                                              bool* pfrh_passTopoThresh,
                                              int* pcrhfracind,
                                              float* pcrhfrac,
                                              float* fracSum,
                                              int* rhCount,
                                              int* topoSeedCount,
                                              int* topoRHCount,
                                              int* seedFracOffsets,
                                              int* topoSeedOffsets,
                                              int* topoSeedList,
                                              float4* pfc_pos4,
                                              float4* pfc_prevPos4,
                                              float4* pfc_linearPos4,
                                              float4* pfc_convPos4,
                                              float* pfc_energy,
                                              float* pfc_clusterT0,
                                              float (&timer)[8],
                                              int* topoIter,
                                              int* pfcIter,
                                              int* pcrhFracSize) {
    if (nRH < 1)
      return;
    cudaProfilerStart();

#ifdef DEBUG_GPU_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStream);
#endif
    // Combined seeding & topo clustering thresholds
    seedingTopoThreshKernel_ECAL<<<(nRH + 63) / 64, 128, 0, cudaStream>>>(nRH,
                                                                          fracSum,
                                                                          pfrh_energy,
                                                                          pfrh_pt2,
                                                                          pfrh_isSeed,
                                                                          pfrh_topoId,
                                                                          pfrh_passTopoThresh,
                                                                          pfrh_layer,
                                                                          neigh8_Ind,
                                                                          rhCount,
                                                                          topoSeedCount,
                                                                          topoRHCount,
                                                                          seedFracOffsets,
                                                                          topoSeedOffsets,
                                                                          topoSeedList,
                                                                          pfcIter);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaEventRecord(start, cudaStream);
#endif

    //topoclustering
    //topoClusterLinking<<<1, 1024 >>>(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, topoIter);
    topoClusterLinking<<<1, 512, 0, cudaStream>>>(
        nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, topoIter);
    //topoClusterContraction<<<1, 512>>>(nRH, pfrh_topoId);
    topoClusterContraction<<<1, 512, 0, cudaStream>>>(nRH,
                                                      pfrh_topoId,
                                                      pfrh_isSeed,
                                                      rhCount,
                                                      topoSeedCount,
                                                      topoRHCount,
                                                      seedFracOffsets,
                                                      topoSeedOffsets,
                                                      topoSeedList,
                                                      pcrhfracind,
                                                      pcrhfrac,
                                                      pcrhFracSize);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaEventRecord(start, cudaStream);
#endif

    dim3 grid((nRH + 31) / 32, (nRH + 31) / 32);
    dim3 block(32, 32);
    //printf("grid = (%d, %d, %d)\tblock = (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    //printf("About to call fillRhfIndex with nRH = %d\n", nRH);

    //    resetOldFracArrays<<<32, 128>>>(nRH, pcrhfracind, pcrhfrac);
    fillRhfIndex<<<grid, block, 0, cudaStream>>>(
        nRH, pfrh_topoId, pfrh_isSeed, topoSeedCount, topoRHCount, seedFracOffsets, rhCount, pcrhfracind);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaEventRecord(start, cudaStream);
#endif

    fastCluster<<<nRH, 256, 0, cudaStream>>>(nRH,
                                             pfrh_x,
                                             pfrh_y,
                                             pfrh_z,
                                             geomAxis_x,
                                             geomAxis_y,
                                             geomAxis_z,
                                             pfrh_energy,
                                             pfrh_topoId,
                                             pfrh_isSeed,
                                             pfrh_layer,
                                             neigh8_Ind,
                                             pcrhfrac,
                                             pcrhfracind,
                                             fracSum,
                                             rhCount,
                                             topoSeedCount,
                                             topoRHCount,
                                             seedFracOffsets,
                                             topoSeedOffsets,
                                             topoSeedList,
                                             pfc_pos4,
                                             pfc_prevPos4,
                                             pfc_linearPos4,
                                             pfc_convPos4,
                                             pfc_energy,
                                             pfc_clusterT0,
                                             pfcIter);

    //    printf("*** After fastCluster ***\n");
    //    printFracs<<<1,1>>>(nRH, pcrhfrac, pcrhfracind, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList);
    //

    //printf("About to run printPFCIter\n");
    //printPFCIter<<<1,1>>>(nRH, pfcIter);

    //fastCluster_serialize<<<1, 1>>>(nRH, pfrh_x,  pfrh_y,  pfrh_z,  geomAxis_x, geomAxis_y, geomAxis_z, pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, neigh8_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount);

    //fastCluster_original<<<1, 1>>>(nRH, pfrh_x,  pfrh_y,  pfrh_z,  geomAxis_x, geomAxis_y, geomAxis_z, pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);

/*
    dim3 grid2( (nRH+32-1)/32, (nRH+32-1)/32 );
    dim3 block2( 32, 32);

    fastCluster_step1<<<grid2, block2>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaEventRecord(start);
#endif

    fastCluster_step2<<<grid2, block2>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
*/
#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[3], start, stop);
#endif
    cudaProfilerStop();
  }

  void PFRechitToPFCluster_ECALV2(size_t size,
                                  const float* __restrict__ pfrh_x,
                                  const float* __restrict__ pfrh_y,
                                  const float* __restrict__ pfrh_z,
                                  const float* __restrict__ pfrh_energy,
                                  const float* __restrict__ pfrh_pt2,
                                  int* pfrh_isSeed,
                                  int* pfrh_topoId,
                                  const int* __restrict__ pfrh_layer,
                                  const int* __restrict__ neigh8_Ind,
                                  int* pcrhfracind,
                                  float* pcrhfrac,
                                  float* fracSum,
                                  int* rhCount,
                                  float (&timer)[8]) {
#ifdef DEBUG_GPU_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    cudaMemsetAsync(rhCount, 1, sizeof(int) * size);
    cudaMemsetAsync(fracSum, 0., sizeof(float) * size);
    //seeding
    if (size > 0)
      seedingKernel_ECAL<<<(size + 512 - 1) / 512, 512>>>(
          size, pfrh_energy, pfrh_pt2, pfrh_isSeed, pfrh_topoId, pfrh_layer, neigh8_Ind);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaEventRecord(start);
#endif

    // for(int a=0;a<16;a++){
    //if(size>0) topoKernel_ECAL<<<(size+512-1)/512, 512>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    //}

    dim3 gridT((size + 64 - 1) / 64, 1);
    dim3 blockT(64, 8);
    //dim3 gridT( (size+64-1)/64, 8 );
    //dim3 blockT( 64, 16);
    //for(int h=0; h<18; h++){
    for (int h = 0; h < nTopoLoops; h++) {
      if (size > 0)
        topoKernel_ECALV2<<<gridT, blockT>>>(size, pfrh_energy, pfrh_topoId, pfrh_layer, neigh8_Ind);
    }

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[1], start, stop);

    cudaEventRecord(start);
#endif
    dim3 grid((size + 32 - 1) / 32, (size + 32 - 1) / 32);
    dim3 block(32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

    if (size > 0)
      fastCluster_step1<<<grid, block>>>(size,
                                         pfrh_x,
                                         pfrh_y,
                                         pfrh_z,
                                         pfrh_energy,
                                         pfrh_topoId,
                                         pfrh_isSeed,
                                         pfrh_layer,
                                         pcrhfrac,
                                         pcrhfracind,
                                         fracSum,
                                         rhCount);
#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[2], start, stop);

    cudaEventRecord(start);
#endif

    if (size > 0)
      fastCluster_step2<<<grid, block>>>(size,
                                         pfrh_x,
                                         pfrh_y,
                                         pfrh_z,
                                         pfrh_energy,
                                         pfrh_topoId,
                                         pfrh_isSeed,
                                         pfrh_layer,
                                         pcrhfrac,
                                         pcrhfracind,
                                         fracSum,
                                         rhCount);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[3], start, stop);
#endif
  }

  void PFRechitToPFCluster_ECAL_serialize(size_t size,
                                          const float* __restrict__ pfrh_x,
                                          const float* __restrict__ pfrh_y,
                                          const float* __restrict__ pfrh_z,
                                          const float* __restrict__ pfrh_energy,
                                          const float* __restrict__ pfrh_pt2,
                                          int* pfrh_isSeed,
                                          int* pfrh_topoId,
                                          const int* __restrict__ pfrh_layer,
                                          const int* __restrict__ neigh8_Ind,
                                          int* pcrhfracind,
                                          float* pcrhfrac,
                                          float* fracSum,
                                          int* rhCount,
                                          float* timer) {
#ifdef DEBUG_GPU_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
    //seeding
    if (size > 0)
      seedingKernel_ECAL_serialize<<<1, 1>>>(
          size, pfrh_energy, pfrh_pt2, pfrh_isSeed, pfrh_topoId, pfrh_layer, neigh8_Ind);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(start);
#endif
    for (int h = 0; h < nTopoLoops; h++) {
      if (size > 0)
        topoKernel_ECAL_serialize<<<1, 1>>>(size, pfrh_energy, pfrh_topoId, pfrh_layer, neigh8_Ind);
    }

#ifdef DEBUG_GPU_ECAL
    float milliseconds = 0;
    if (timer != NULL) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      *timer = milliseconds;
    }
#endif
    //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    //dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

    if (size > 0)
      fastCluster_step1_serialize<<<1, 1>>>(size,
                                            pfrh_x,
                                            pfrh_y,
                                            pfrh_z,
                                            pfrh_energy,
                                            pfrh_topoId,
                                            pfrh_isSeed,
                                            pfrh_layer,
                                            pcrhfrac,
                                            pcrhfracind,
                                            fracSum,
                                            rhCount);

    if (size > 0)
      fastCluster_step2_serialize<<<1, 1>>>(size,
                                            pfrh_x,
                                            pfrh_y,
                                            pfrh_z,
                                            pfrh_energy,
                                            pfrh_topoId,
                                            pfrh_isSeed,
                                            pfrh_layer,
                                            pcrhfrac,
                                            pcrhfracind,
                                            fracSum,
                                            rhCount);
  }

}  // namespace PFClusterCudaECAL

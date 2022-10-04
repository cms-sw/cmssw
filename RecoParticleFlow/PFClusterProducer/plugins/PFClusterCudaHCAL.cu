#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Eigen include files
#include <Eigen/Dense>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "PFClusterCudaHCAL.h"

using PFClustering::common::PFLayer;

// Uncomment for debugging
//#define DEBUG_GPU_HCAL

constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);
constexpr const float PI_F = 3.141592654f;

namespace PFClusterCudaHCAL {

  __constant__ float showerSigma2;
  __constant__ float recHitEnergyNormInvEB_vec[4];
  __constant__ float recHitEnergyNormInvEE_vec[7];
  __constant__ float minFracToKeep;
  __constant__ float minFracTot;
  __constant__ float minFractionInCalc;
  __constant__ float minAllowedNormalization;
  __constant__ float stoppingTolerance;

  __constant__ float seedEThresholdEB_vec[4];
  __constant__ float seedEThresholdEE_vec[7];
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB_vec[4];
  __constant__ float topoEThresholdEE_vec[7];
  __constant__ int maxIterations;
  __constant__ bool excludeOtherSeeds;

  // Endcap timing constants
  __constant__ float corrTermLowEE;
  __constant__ float threshLowEE;
  __constant__ float noiseTermE;
  __constant__ float constantTermLowE2E;
  __constant__ float noiseTermLowEE;
  __constant__ float threshHighEE;
  __constant__ float constantTerm2E;
  __constant__ float resHighE2E;

  // Barrel timing constants
  __constant__ float corrTermLowEB;
  __constant__ float threshLowEB;
  __constant__ float noiseTermB;
  __constant__ float constantTermLowE2B;
  __constant__ float noiseTermLowEB;
  __constant__ float threshHighEB;
  __constant__ float constantTerm2B;
  __constant__ float resHighE2B;

  __constant__ int nNT = 8;  // Number of neighbors considered for topo clustering
  __constant__ int nNeigh;

  //int nTopoLoops = 100;
  int nTopoLoops = 35;

  //
  // --- kernel summary --
  // initializeCudaConstants
  // PFRechitToPFCluster_HCAL_entryPoint
  //   seedingTopoThreshKernel_HCAL: apply seeding/topo-clustering threshold to RecHits, also ensure a peak (outputs: pfrh_isSeed, pfrh_passTopoThresh) [OutputDataGPU]
  //   prepareTopoInputs: prepare "edge" data (outputs: nEdges, pfrh_edgeId, pfrh_edgeList [nEdges dimension])
  //   topoClusterLinking: run topo clustering (output: pfrh_topoId)
  //   topoClusterContraction: find parent of parent (or parent (of parent ...)) (outputs: pfrh_parent, topoSeedCount, topoSeedOffsets, topoSeedList, seedFracOffsets, pcrhfracind, pcrhfrac)
  //   fillRhfIndex: fill rhfracind (PFCluster RecHitFraction constituent PFRecHit indices)
  //   hcalFastCluster_selection
  //     dev_hcalFastCluster_optimizedSimple
  //     dev_hcalFastCluster_optimizedComplex
  //     dev_hcalFastCluster_original
  // [aux]
  //     sortEight
  //     sortSwap
  // [not used]
  //   (fillRhfIndex_serialize) serialized version
  //   (prepareTopoInputsSerial) serialized version
  //   [compareEdgeArrays] used only for debugging
  //   seedingKernel_HCAL
  //   seedingKernel_HCAL_serialize
  //   compareEdgeArrays
  //   topoKernel_HCAL_passTopoThresh
  //   topoKernel_HCALV2
  //   topoKernel_HCAL_serialize
  //   hcalFastCluster_optimizedSimple
  //   hcalFastCluster_optimizedComplex
  //   hcalFastCluster_sharedRHList
  //   hcalFastCluster_original
  //   hcalFastCluster_serialize
  //   hcalFastCluster_step1
  //   hcalFastCluster_step2
  //   hcalFastCluster_step2
  //   hcalFastCluster_step1_serialize
  //   hcalFastCluster_step2_serialize
  //   passingTopoThreshold
  //   passingTopoThreshold
  //   printRhfIndex

  void initializeCudaConstants(const PFClustering::common::CudaHCALConstants& cudaConstants,
                               const cudaStream_t cudaStream) {
    cudaCheck(cudaMemcpyToSymbolAsync(
        showerSigma2, &cudaConstants.showerSigma2, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    std::cout << "--- HCAL Cuda constant values ---" << std::endl;
    float val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, showerSigma2, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "showerSigma2 read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormInvEB_vec,
                                      &cudaConstants.recHitEnergyNormInvEB_vec,
                                      4 * sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    float val4[4];
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val4, recHitEnergyNormInvEB_vec, 4 * sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "recHitEnergyNormInvEB_vec read from symbol: ";
    for (int i = 0; i < 4; i++) {
      std::cout << val4[i] << " ";
    }
    std::cout << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormInvEE_vec,
                                      &cudaConstants.recHitEnergyNormInvEE_vec,
                                      7 * sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    float val7[7];
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val7, recHitEnergyNormInvEE_vec, 7 * sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "recHitEnergyNormInvEE_vec read from symbol: ";
    for (int i = 0; i < 7; i++) {
      std::cout << val7[i] << " ";
    }
    std::cout << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        minFracToKeep, &cudaConstants.minFracToKeep, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, minFracToKeep, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minFracToKeep read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        minFracTot, &cudaConstants.minFracTot, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, minFracTot, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minFracTot read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        minFractionInCalc, &cudaConstants.minFracInCalc, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, minFractionInCalc, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minFractionInCalc read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(minAllowedNormalization,
                                      &cudaConstants.minAllowedNormalization,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(
        cudaMemcpyFromSymbolAsync(&val, minAllowedNormalization, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "minAllowedNormalization read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        stoppingTolerance, &cudaConstants.stoppingTolerance, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, stoppingTolerance, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "stoppingTolerance read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        excludeOtherSeeds, &cudaConstants.excludeOtherSeeds, sizeof(bool), 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    bool bval = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&bval, excludeOtherSeeds, sizeof(bool), 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "excludeOtherSeeds read from symbol: " << bval << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        maxIterations, &cudaConstants.maxIterations, sizeof_int, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    uint32_t uival = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&uival, maxIterations, sizeof_int, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "maxIterations read from symbol: " << uival << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEB_vec,
                                      &cudaConstants.seedEThresholdEB_vec,
                                      4 * sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val4, seedEThresholdEB_vec, 4 * sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedEThresholdEB_vec read from symbol: ";
    for (int i = 0; i < 4; i++) {
      std::cout << val4[i] << " ";
    }
    std::cout << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEE_vec,
                                      &cudaConstants.seedEThresholdEE_vec,
                                      7 * sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val7, seedEThresholdEE_vec, 7 * sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedEThresholdEE_vec read from symbol: ";
    for (int i = 0; i < 7; i++) {
      std::cout << val7[i] << " ";
    }
    std::cout << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        seedPt2ThresholdEB, &cudaConstants.seedPt2ThresholdEB, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, seedPt2ThresholdEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedPt2ThresholdEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        seedPt2ThresholdEE, &cudaConstants.seedPt2ThresholdEE, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0.;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, seedPt2ThresholdEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "seedPt2ThresholdEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEB_vec,
                                      &cudaConstants.topoEThresholdEB_vec,
                                      4 * sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val4, topoEThresholdEB_vec, 4 * sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "topoEThresholdEB_vec read from symbol: ";
    for (int i = 0; i < 4; i++) {
      std::cout << val4[i] << " ";
    }
    std::cout << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEE_vec,
                                      &cudaConstants.topoEThresholdEE_vec,
                                      7 * sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    cudaCheck(cudaMemcpyFromSymbolAsync(
        &val7, topoEThresholdEE_vec, 7 * sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "topoEThresholdEE_vec read from symbol: ";
    for (int i = 0; i < 7; i++) {
      std::cout << val7[i] << " ";
    }
    std::cout << std::endl;
#endif

    cudaCheck(
        cudaMemcpyToSymbolAsync(nNeigh, &cudaConstants.nNeigh, sizeof_int, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    int ival = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&ival, nNeigh, sizeof_int, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "nNeigh read from symbol: " << ival << std::endl;
#endif

    // Endcap time resolution
    cudaCheck(cudaMemcpyToSymbolAsync(corrTermLowEE,
                                      &cudaConstants.endcapTimeResConsts.corrTermLowE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, corrTermLowEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "corrTermLowEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(threshLowEE,
                                      &cudaConstants.endcapTimeResConsts.threshLowE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, threshLowEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "threshLowEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        noiseTermE, &cudaConstants.endcapTimeResConsts.noiseTerm, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, noiseTermE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "noiseTermE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(constantTermLowE2E,
                                      &cudaConstants.endcapTimeResConsts.constantTermLowE2,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, constantTermLowE2E, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "constantTermLowE2E read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(noiseTermLowEE,
                                      &cudaConstants.endcapTimeResConsts.noiseTermLowE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, noiseTermLowEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "noiseTermLowEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(threshHighEE,
                                      &cudaConstants.endcapTimeResConsts.threshHighE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, threshHighEE, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "threshHighEE read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(constantTerm2E,
                                      &cudaConstants.endcapTimeResConsts.constantTerm2,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, constantTerm2E, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "constantTerm2E read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        resHighE2E, &cudaConstants.endcapTimeResConsts.resHighE2, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, resHighE2E, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "resHighE2E read from symbol: " << val << std::endl;
#endif

    // Barrel time resolution
    cudaCheck(cudaMemcpyToSymbolAsync(corrTermLowEB,
                                      &cudaConstants.barrelTimeResConsts.corrTermLowE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, corrTermLowEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "corrTermLowEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(threshLowEB,
                                      &cudaConstants.barrelTimeResConsts.threshLowE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, threshLowEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "threshLowEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        noiseTermB, &cudaConstants.barrelTimeResConsts.noiseTerm, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, noiseTermB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "noiseTermB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(constantTermLowE2B,
                                      &cudaConstants.barrelTimeResConsts.constantTermLowE2,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, constantTermLowE2B, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "constantTermLowE2B read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(noiseTermLowEB,
                                      &cudaConstants.barrelTimeResConsts.noiseTermLowE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, noiseTermLowEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "noiseTermLowEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(threshHighEB,
                                      &cudaConstants.barrelTimeResConsts.threshHighE,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, threshHighEB, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "threshHighEB read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(constantTerm2B,
                                      &cudaConstants.barrelTimeResConsts.constantTerm2,
                                      sizeof_float,
                                      0,
                                      cudaMemcpyHostToDevice,
                                      cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, constantTerm2B, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "constantTerm2B read from symbol: " << val << std::endl;
#endif

    cudaCheck(cudaMemcpyToSymbolAsync(
        resHighE2B, &cudaConstants.barrelTimeResConsts.resHighE2, sizeof_float, 0, cudaMemcpyHostToDevice, cudaStream));
#ifdef DEBUG_GPU_HCAL
    // Read back the value
    val = 0;
    cudaCheck(cudaMemcpyFromSymbolAsync(&val, resHighE2B, sizeof_float, 0, cudaMemcpyDeviceToHost, cudaStream));
    std::cout << "resHighE2B read from symbol: " << val << std::endl;
#endif
  }

  __device__ __forceinline__ float timeResolution2Endcap(const float energy) {
    float res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < threshLowEE) {
      if (corrTermLowEE > 0.) {  // different parametrisation
        const float res = noiseTermLowEE / energy + corrTermLowEE / (energy * energy);
        res2 = res * res;
      } else {
        const float noiseDivE = noiseTermLowEE / energy;
        res2 = noiseDivE * noiseDivE + constantTermLowE2E;
      }
    } else if (energy < threshHighEE) {
      const float noiseDivE = noiseTermE / energy;
      res2 = noiseDivE * noiseDivE + constantTerm2E;
    } else  // if (energy >=threshHighE_)
      res2 = resHighE2E;

    if (res2 > 10000.)
      return 10000.;
    return res2;
  }

  __device__ __forceinline__ float timeResolution2Barrel(const float energy) {
    float res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < threshLowEB) {
      if (corrTermLowEB > 0.) {  // different parametrisation
        const float res = noiseTermLowEB / energy + corrTermLowEB / (energy * energy);
        res2 = res * res;
      } else {
        const float noiseDivE = noiseTermLowEB / energy;
        res2 = noiseDivE * noiseDivE + constantTermLowE2B;
      }
    } else if (energy < threshHighEB) {
      const float noiseDivE = noiseTermB / energy;
      res2 = noiseDivE * noiseDivE + constantTerm2B;
    } else  // if (energy >=threshHighE_)
      res2 = resHighE2B;

    if (res2 > 10000.)
      return 10000.;
    return res2;
  }

  __device__ __forceinline__ float dR2(float4 pos1, float4 pos2) {
    float mag1 = sqrtf(pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
    float cosTheta1 = mag1 > 0.0 ? pos1.z / mag1 : 1.0;
    float eta1 = 0.5 * logf((1.0 + cosTheta1) / (1.0 - cosTheta1));
    float phi1 = atan2f(pos1.y, pos1.x);

    float mag2 = sqrtf(pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
    float cosTheta2 = mag2 > 0.0 ? pos2.z / mag2 : 1.0;
    float eta2 = 0.5 * logf((1.0 + cosTheta2) / (1.0 - cosTheta2));
    float phi2 = atan2f(pos2.y, pos2.x);

    float deta = eta2 - eta1;
    float dphi = fabsf(fabsf(phi2 - phi1) - PI_F) - PI_F;
    return (deta * deta + dphi * dphi);
  }

  // https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
  /*static __device__ __forceinline__ float atomicMinF(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}*/

  static __device__ __forceinline__ float atomicMaxF(float* address, float val) {
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret)) {
      int old = ret;
      if ((ret = atomicCAS((int*)address, old, __float_as_int(val))) == old)
        break;
    }
    return __int_as_float(ret);
  }

  __global__ void seedingTopoThreshKernel_HCAL(size_t size,
                                               const float* __restrict__ pfrh_energy,
                                               const float* __restrict__ pfrh_x,
                                               const float* __restrict__ pfrh_y,
                                               const float* __restrict__ pfrh_z,
                                               int* pfrh_isSeed,
                                               int* pfrh_topoId,
                                               //bool*  pfrh_passTopoThresh,
                                               int* pfrh_passTopoThresh,
                                               const int* __restrict__ pfrh_layer,
                                               const int* __restrict__ pfrh_depth,
                                               const int* __restrict__ neigh4_Ind,
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
      topoSeedCount[i] = 0;
      topoRHCount[i] = 0;
      seedFracOffsets[i] = -1;
      topoSeedOffsets[i] = -1;
      topoSeedList[i] = -1;
      pfcIter[i] = -1;

      int layer = pfrh_layer[i];
      int depthOffset = pfrh_depth[i] - 1;
      float energy = pfrh_energy[i];
      float3 pos = make_float3(pfrh_x[i], pfrh_y[i], pfrh_z[i]);

      // cmssdt.cern.ch/lxr/source/DataFormats/ParticleFlowReco/interface/PFRecHit.h#0108
      float pt2 = energy * energy * (pos.x * pos.x + pos.y * pos.y) / (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

      // Seeding threshold test
      if ((layer == PFLayer::HCAL_BARREL1 && energy > seedEThresholdEB_vec[depthOffset] && pt2 > seedPt2ThresholdEB) ||
          (layer == PFLayer::HCAL_ENDCAP && energy > seedEThresholdEE_vec[depthOffset] && pt2 > seedPt2ThresholdEE)) {
        pfrh_isSeed[i] = 1;
        for (int k = 0; k < 4; k++) {
          if (neigh4_Ind[8 * i + k] < 0)
            continue;
          if (energy < pfrh_energy[neigh4_Ind[8 * i + k]]) {
            pfrh_isSeed[i] = 0;
            //pfrh_topoId[i]=-1;
            break;
          }
        }
        //         for(int k=0; k<nNeigh; k++){
        //           if(neigh4_Ind[nNeigh*i+k]<0) continue;
        //           if(energy < pfrh_energy[neigh4_Ind[nNeigh*i+k]]){
        //             pfrh_isSeed[i]=0;
        //             //pfrh_topoId[i]=-1;
        //             break;
        //           }
        //	     }
      } else {
        // pfrh_topoId[i]=-1;
        pfrh_isSeed[i] = 0;
      }

      // Topo clustering threshold test
      if ((layer == PFLayer::HCAL_ENDCAP && energy > topoEThresholdEE_vec[depthOffset]) ||
          (layer == PFLayer::HCAL_BARREL1 && energy > topoEThresholdEB_vec[depthOffset])) {
        pfrh_passTopoThresh[i] = true;
      }
      //else { pfrh_passTopoThresh[i] = false; }
      else {
        pfrh_passTopoThresh[i] = false;
        pfrh_topoId[i] = -1;
      }
    }
  }
  __global__ void seedingKernel_HCAL(size_t size,
                                     const float* __restrict__ pfrh_energy,
                                     const float* __restrict__ pfrh_pt2,
                                     int* pfrh_isSeed,
                                     int* pfrh_topoId,
                                     const int* __restrict__ pfrh_layer,
                                     const int* __restrict__ pfrh_depth,
                                     const int* __restrict__ neigh4_Ind) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size) {
      if ((pfrh_layer[i] == PFLayer::HCAL_BARREL1 && pfrh_energy[i] > seedEThresholdEB_vec[pfrh_depth[i] - 1] &&
           pfrh_pt2[i] > seedPt2ThresholdEB) ||
          (pfrh_layer[i] == PFLayer::HCAL_ENDCAP && pfrh_energy[i] > seedEThresholdEE_vec[pfrh_depth[i] - 1] &&
           pfrh_pt2[i] > seedPt2ThresholdEE)) {
        pfrh_isSeed[i] = 1;
        for (int k = 0; k < nNeigh; k++) {
          if (neigh4_Ind[nNeigh * i + k] < 0)
            continue;
          if (pfrh_energy[i] < pfrh_energy[neigh4_Ind[nNeigh * i + k]]) {
            pfrh_isSeed[i] = 0;
            //pfrh_topoId[i]=-1;
            break;
          }
        }
      } else {
        // pfrh_topoId[i]=-1;
        pfrh_isSeed[i] = 0;
      }
    }
  }

  __global__ void seedingKernel_HCAL_serialize(size_t size,
                                               const float* __restrict__ pfrh_energy,
                                               const float* __restrict__ pfrh_pt2,
                                               int* pfrh_isSeed,
                                               int* pfrh_topoId,
                                               const int* __restrict__ pfrh_layer,
                                               const int* __restrict__ pfrh_depth,
                                               const int* __restrict__ neigh4_Ind) {
    //int i = threadIdx.x+blockIdx.x*blockDim.x;
    for (int i = 0; i < size; i++) {
      if (i < size) {
        if ((pfrh_layer[i] == PFLayer::HCAL_BARREL1 && pfrh_energy[i] > seedEThresholdEB_vec[pfrh_depth[i] - 1] &&
             pfrh_pt2[i] > seedPt2ThresholdEB) ||
            (pfrh_layer[i] == PFLayer::HCAL_ENDCAP && pfrh_energy[i] > seedEThresholdEE_vec[pfrh_depth[i] - 1] &&
             pfrh_pt2[i] > seedPt2ThresholdEE)) {
          pfrh_isSeed[i] = 1;
          for (int k = 0; k < nNeigh; k++) {
            if (neigh4_Ind[nNeigh * i + k] < 0)
              continue;
            if (pfrh_energy[i] < pfrh_energy[neigh4_Ind[nNeigh * i + k]]) {
              pfrh_isSeed[i] = 0;
              //pfrh_topoId[i]=-1;
              break;
            }
          }
        } else {
          // pfrh_topoId[i]=-1;
          pfrh_isSeed[i] = 0;
        }
      }
    }
  }

  __global__ void topoKernel_HCAL_passTopoThresh(size_t size,
                                                 const float* __restrict__ pfrh_energy,
                                                 int* pfrh_topoId,
                                                 const bool* __restrict__ pfrh_passTopoThresh,
                                                 const int* __restrict__ neigh8_Ind) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    int k = (threadIdx.y + blockIdx.y * blockDim.y) % nNT;

    //if(l<size && k<nNT) {
    if (l < size) {
      while (pfrh_passTopoThresh[nNT * l + k] && neigh8_Ind[nNT * l + k] > -1 &&
             pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
        if (pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
          atomicMax(&pfrh_topoId[neigh8_Ind[nNT * l + k]], pfrh_topoId[l]);
        }
        if (pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
          atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNT * l + k]]);
        }
      }
    }
  }

  __global__ void topoKernel_HCALV2(size_t size,
                                    const float* __restrict__ pfrh_energy,
                                    int* pfrh_topoId,
                                    const int* __restrict__ pfrh_layer,
                                    const int* __restrict__ pfrh_depth,
                                    const int* __restrict__ neigh8_Ind) {
    int l = threadIdx.x + blockIdx.x * blockDim.x;
    //int k = threadIdx.y+blockIdx.y*blockDim.y;
    int k = (threadIdx.y + blockIdx.y * blockDim.y) % nNT;

    //if(l<size && k<nNT) {
    if (l < size) {
      while (neigh8_Ind[nNT * l + k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNT * l + k]] &&
             ((pfrh_layer[neigh8_Ind[nNT * l + k]] == PFLayer::HCAL_ENDCAP &&
               pfrh_energy[neigh8_Ind[nNT * l + k]] > topoEThresholdEE_vec[pfrh_depth[neigh8_Ind[nNT * l + k]] - 1]) ||
              (pfrh_layer[neigh8_Ind[nNT * l + k]] == PFLayer::HCAL_BARREL1 &&
               pfrh_energy[neigh8_Ind[nNT * l + k]] > topoEThresholdEB_vec[pfrh_depth[neigh8_Ind[nNT * l + k]] - 1])) &&
             ((pfrh_layer[l] == PFLayer::HCAL_ENDCAP && pfrh_energy[l] > topoEThresholdEE_vec[pfrh_depth[l] - 1]) ||
              (pfrh_layer[l] == PFLayer::HCAL_BARREL1 && pfrh_energy[l] > topoEThresholdEB_vec[pfrh_depth[l] - 1]))) {
        if (pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
          atomicMax(&pfrh_topoId[neigh8_Ind[nNT * l + k]], pfrh_topoId[l]);
        }
        if (pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
          atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNT * l + k]]);
        }
      }
    }
  }

  __global__ void topoKernel_HCAL_serialize(size_t size,
                                            const float* __restrict__ pfrh_energy,
                                            int* pfrh_topoId,
                                            const int* __restrict__ pfrh_layer,
                                            const int* __restrict__ pfrh_depth,
                                            const int* __restrict__ neigh8_Ind) {
    //int l = threadIdx.x+blockIdx.x*blockDim.x;
    //int k = threadIdx.y+blockIdx.y*blockDim.y;

    for (int l = 0; l < size; l++) {
      //for (int k = 0; k < size; k++) {
      for (int k = 0; k < 8; k++) {
        while (
            neigh8_Ind[nNT * l + k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNT * l + k]] &&
            ((pfrh_layer[neigh8_Ind[nNT * l + k]] == PFLayer::HCAL_ENDCAP &&
              pfrh_energy[neigh8_Ind[nNT * l + k]] > topoEThresholdEE_vec[pfrh_depth[neigh8_Ind[nNT * l + k]] - 1]) ||
             (pfrh_layer[neigh8_Ind[nNT * l + k]] == PFLayer::HCAL_BARREL1 &&
              pfrh_energy[neigh8_Ind[nNT * l + k]] > topoEThresholdEB_vec[pfrh_depth[neigh8_Ind[nNT * l + k]] - 1])) &&
            ((pfrh_layer[l] == PFLayer::HCAL_ENDCAP && pfrh_energy[l] > topoEThresholdEE_vec[pfrh_depth[l] - 1]) ||
             (pfrh_layer[l] == PFLayer::HCAL_BARREL1 && pfrh_energy[l] > topoEThresholdEB_vec[pfrh_depth[l] - 1]))) {
          if (pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
            atomicMax(&pfrh_topoId[neigh8_Ind[nNT * l + k]], pfrh_topoId[l]);
          }
          if (pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNT * l + k]]) {
            atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNT * l + k]]);
          }
        }
      }
    }
  }

  __device__ void dev_hcalFastCluster_optimizedSimple(int topoId,
                                                      int nRHTopo,
                                                      const float* __restrict__ pfrh_x,
                                                      const float* __restrict__ pfrh_y,
                                                      const float* __restrict__ pfrh_z,
                                                      const float* __restrict__ pfrh_energy,
                                                      const int* __restrict__ pfrh_layer,
                                                      const int* __restrict__ pfrh_depth,
                                                      float* pcrhfrac,
                                                      int* pcrhfracind,
                                                      int* topoSeedOffsets,
                                                      int* topoSeedList,
                                                      int* seedFracOffsets,
                                                      int* pfcIter) {
    int tid = threadIdx.x;  // thread index is rechit number
    __shared__ int i, iter, nRHOther;
    __shared__ float tol, clusterEnergy, rhENormInv, seedEnergy;
    __shared__ float4 clusterPos, prevClusterPos, seedPos;
    __shared__ bool notDone, debug;
    if (tid == 0) {
      i = topoSeedList[topoSeedOffsets[topoId]];  // i is the seed rechit index
      nRHOther = nRHTopo - 1;
      seedPos = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.);
      clusterPos = seedPos;
      prevClusterPos = seedPos;
      seedEnergy = pfrh_energy[i];
      clusterEnergy = seedEnergy;
      tol = stoppingTolerance;  // stopping tolerance * tolerance scaling

      if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
        rhENormInv = recHitEnergyNormInvEB_vec[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = recHitEnergyNormInvEE_vec[pfrh_depth[i] - 1];
      else {
        rhENormInv = 0.;
        printf("Rechit %d has invalid layer %d!\n", i, pfrh_layer[i]);
      }

      iter = 0;
      notDone = true;
      debug = false;
      //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
    }
    __syncthreads();

    int j = -1;  // j is the rechit index for this thread
    int rhFracOffset = -1;
    float4 rhPos;
    float rhEnergy = -1., rhPosNorm = -1.;

    if (tid < nRHOther) {
      rhFracOffset = seedFracOffsets[i] + tid + 1;  // Offset for this rechit in pcrhfrac, pcrhfracidx arrays
      j = pcrhfracind[rhFracOffset];                // rechit index for this thread
      rhPos = make_float4(pfrh_x[j], pfrh_y[j], pfrh_z[j], 1.);
      rhEnergy = pfrh_energy[j];
      rhPosNorm = fmaxf(0., logf(rhEnergy * rhENormInv));
    }
    __syncthreads();

    do {
      if (debug && tid == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }
      float dist2 = -1., d2 = -1., fraction = -1.;
      if (tid < nRHOther) {
        dist2 = (clusterPos.x - rhPos.x) * (clusterPos.x - rhPos.x) +
                (clusterPos.y - rhPos.y) * (clusterPos.y - rhPos.y) +
                (clusterPos.z - rhPos.z) * (clusterPos.z - rhPos.z);

        d2 = dist2 / showerSigma2;
        fraction = clusterEnergy * rhENormInv * expf(-0.5 * d2);

        // For single seed clusters, rechit fraction is either 1 (100%) or -1 (not included)
        if (fraction > minFracTot && d2 < 100.)
          fraction = 1.;
        else
          fraction = -1.;
        pcrhfrac[rhFracOffset] = fraction;
      }
      __syncthreads();

      if (debug && tid == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      if (tid == 0) {
        // Reset cluster position and energy
        clusterPos = seedPos;
        clusterEnergy = seedEnergy;
      }
      __syncthreads();

      // Recalculate cluster position and energy
      if (fraction > -0.5) {
        atomicAdd(&clusterEnergy, rhEnergy);
        //computeClusterPos(clusterPos, rechitPos, rhEnergy, rhENormInv, debug);
        atomicAdd(&clusterPos.x, rhPos.x * rhPosNorm);
        atomicAdd(&clusterPos.y, rhPos.y * rhPosNorm);
        atomicAdd(&clusterPos.z, rhPos.z * rhPosNorm);
        atomicAdd(&clusterPos.w, rhPosNorm);  // position_norm
      }
      __syncthreads();

      if (tid == 0) {
        // Normalize the seed postiion
        if (clusterPos.w >= minAllowedNormalization) {
          // Divide by position norm
          clusterPos.x /= clusterPos.w;
          clusterPos.y /= clusterPos.w;
          clusterPos.z /= clusterPos.w;

          if (debug)
            printf("\tPF cluster (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   i,
                   clusterEnergy,
                   clusterPos.x,
                   clusterPos.y,
                   clusterPos.z);
        } else {
          if (debug)
            printf("\tPF cluster (seed %d) position norm (%f) less than minimum (%f)\n",
                   i,
                   clusterPos.w,
                   minAllowedNormalization);
          clusterPos.x = 0.;
          clusterPos.y = 0.;
          clusterPos.z = 0.;
        }
        float diff2 = dR2(prevClusterPos, clusterPos);
        if (debug)
          printf("\tPF cluster (seed %d) has diff2 = %f\n", i, diff2);
        prevClusterPos = clusterPos;  // Save clusterPos

        float diff = sqrtf(diff2);
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
    } while (notDone);
    if (tid == 0)
      pfcIter[topoId] = iter;
  }

  __device__ void dev_hcalFastCluster_optimizedComplex(int topoId,
                                                       int nSeeds,
                                                       int nRHTopo,
                                                       const float* __restrict__ pfrh_x,
                                                       const float* __restrict__ pfrh_y,
                                                       const float* __restrict__ pfrh_z,
                                                       const float* __restrict__ pfrh_energy,
                                                       const int* __restrict__ pfrh_layer,
                                                       const int* __restrict__ pfrh_depth,
                                                       const int* __restrict__ pfrh_neighbours,
                                                       float* pcrhfrac,
                                                       int* pcrhfracind,
                                                       int* seedFracOffsets,
                                                       int* topoSeedOffsets,
                                                       int* topoSeedList,
                                                       int* pfcIter) {
    int tid = threadIdx.x;

    //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
    __shared__ int nRHNotSeed, topoSeedBegin, gridStride, iter;
    __shared__ float tol, diff2, rhENormInv;
    __shared__ bool notDone, debug;
    __shared__ float4 clusterPos[100], prevClusterPos[100];
    __shared__ float clusterEnergy[100], rhFracSum[256];
    __shared__ int seeds[100], rechits[256];

    if (threadIdx.x == 0) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = topoSeedOffsets[topoId];
      tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
      //gridStride = blockDim.x * gridDim.x;
      gridStride = blockDim.x;
      iter = 0;
      notDone = true;
      debug = false;
      //debug = true;
      //debug = (topoId == 4) ? true : false;
      //debug = (nSeeds == 2 && ( (topoSeedList[topoSeedBegin]==11 && topoSeedList[topoSeedBegin+1]==5) || (topoSeedList[topoSeedBegin]==5 && topoSeedList[topoSeedBegin+1]==11) )) ? true : false;
      //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
      //debug = (topoId == 1 || topoId == 5 || topoId == 6 || topoId == 8 || topoId == 9 || topoId == 10 || topoId == 12 || topoId == 13) ? true : false;

      int i = topoSeedList[topoSeedBegin];
      if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
        rhENormInv = recHitEnergyNormInvEB_vec[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = recHitEnergyNormInvEE_vec[pfrh_depth[i] - 1];
      else
        printf("Rechit %d has invalid layer %d!\n", i, pfrh_layer[i]);
    }
    __syncthreads();

    for (int n = threadIdx.x; n < nRHTopo; n += gridStride) {
      if (n < nSeeds)
        seeds[n] = topoSeedList[topoSeedBegin + n];
      if (n < nRHNotSeed - 1)
        rechits[n] = pcrhfracind[seedFracOffsets[topoSeedList[topoSeedBegin]] + n + 1];
    }
    __syncthreads();

    auto getSeedRhIdx = [&](int seedNum) { return seeds[seedNum]; };

    auto getRhFracIdx = [&](int seedNum, int rhNum) {
      if (rhNum <= 0)
        printf("Invalid rhNum (%d) for getRhFracIdx!\n", rhNum);
      return rechits[rhNum - 1];
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

    auto computeClusterPos = [&](float4& pos4, float frac, int rhInd, bool isDebug) {
      float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
      const auto rh_energy = pfrh_energy[rhInd] * frac;
      const auto norm = (frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
      if (isDebug)
        printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
               rhInd,
               norm,
               frac,
               rh_energy,
               rechitPos.x,
               rechitPos.y,
               rechitPos.z);

      pos4.x += rechitPos.x * norm;
      pos4.y += rechitPos.y * norm;
      pos4.z += rechitPos.z * norm;
      pos4.w += norm;  //  position_norm
    };

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
      int i = getSeedRhIdx(s);
      clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
      prevClusterPos[s] = clusterPos[s];
      clusterEnergy[s] = pfrh_energy[i];
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        pcrhfracind[seedFracOffsets[i] + r + 1] = rechits[r];
        pcrhfrac[seedFracOffsets[i] + r + 1] = -1.;
      }
    }
    __syncthreads();

    int rhThreadIdx = -1;
    float4 rhThreadPos;
    if (tid < (nRHNotSeed - 1)) {
      rhThreadIdx = rechits[tid];  // Index when thread represents rechit
      rhThreadPos = make_float4(pfrh_x[rhThreadIdx], pfrh_y[rhThreadIdx], pfrh_z[rhThreadIdx], 1.);
    }

    // Neighbors when threadIdx represents seed
    int seedThreadIdx = -1;
    int4 seedNeighbors = make_int4(-9, -9, -9, -9);
    float seedEnergy = -1.;
    float4 seedInitClusterPos = make_float4(0., 0., 0., 0.);
    if (tid < nSeeds) {
      seedThreadIdx = getSeedRhIdx(tid);
      seedNeighbors = make_int4(pfrh_neighbours[8 * seedThreadIdx],
                                pfrh_neighbours[8 * seedThreadIdx + 1],
                                pfrh_neighbours[8 * seedThreadIdx + 2],
                                pfrh_neighbours[8 * seedThreadIdx + 3]);
      seedEnergy = pfrh_energy[seedThreadIdx];

      // Compute initial cluster position shift for seed
      computeClusterPos(seedInitClusterPos, 1., seedThreadIdx, debug);
    }

    do {
      if (debug && threadIdx.x == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      // Reset rhFracSum
      rhFracSum[tid] = 0.;
      if (tid == 0)
        diff2 = -1;

      if (tid < (nRHNotSeed - 1)) {
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / showerSigma2;
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);
          //pcrhfrac[seedFracOffsets[seeds[s]]+tid+1] = fraction;

          rhFracSum[tid] += fraction;
        }
      }
      __syncthreads();

      if (tid < (nRHNotSeed - 1)) {
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / showerSigma2;
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > minFracTot) {
            float fracpct = fraction / rhFracSum[tid];
            //float fracpct = pcrhfrac[seedFracOffsets[i]+tid+1] / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
              pcrhfrac[seedFracOffsets[i] + tid + 1] = fracpct;
            } else {
              pcrhfrac[seedFracOffsets[i] + tid + 1] = -1;
            }
          } else {
            pcrhfrac[seedFracOffsets[i] + tid + 1] = -1;
          }
        }
      }
      __syncthreads();

      if (debug && tid == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      // Reset cluster position and energy
      if (tid < nSeeds) {
        clusterPos[tid] = seedInitClusterPos;
        clusterEnergy[tid] = seedEnergy;
        if (debug) {
          printf("Cluster %d (seed %d) has energy %f\tpos = (%f, %f, %f, %f)\n",
                 tid,
                 seeds[tid],
                 clusterEnergy[tid],
                 clusterPos[tid].x,
                 clusterPos[tid].y,
                 clusterPos[tid].z,
                 clusterPos[tid].w);
        }
      }
      __syncthreads();

      // Recalculate position
      if (tid < nSeeds) {
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = getRhFrac(tid, r + 1);

          if (frac > -0.5) {
            clusterEnergy[tid] += frac * pfrh_energy[j];

            if (nSeeds == 1 || j == seedNeighbors.x || j == seedNeighbors.y || j == seedNeighbors.z ||
                j == seedNeighbors.w)
              computeClusterPos(clusterPos[tid], frac, j, debug);
          }
        }
      }
      __syncthreads();

      // Position normalization
      if (tid < nSeeds) {
        if (clusterPos[tid].w >= minAllowedNormalization) {
          // Divide by position norm
          clusterPos[tid].x /= clusterPos[tid].w;
          clusterPos[tid].y /= clusterPos[tid].w;
          clusterPos[tid].z /= clusterPos[tid].w;

          if (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   tid,
                   seedThreadIdx,
                   clusterEnergy[tid],
                   clusterPos[tid].x,
                   clusterPos[tid].y,
                   clusterPos[tid].z);
        } else {
          if (debug)
            printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                   tid,
                   seedThreadIdx,
                   clusterPos[tid].w,
                   minAllowedNormalization);
          clusterPos[tid].x = 0.0;
          clusterPos[tid].y = 0.0;
          clusterPos[tid].z = 0.0;
        }
      }
      __syncthreads();

      if (tid < nSeeds) {
        float delta2 = dR2(prevClusterPos[tid], clusterPos[tid]);
        if (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", tid, seeds[tid], delta2);
        atomicMaxF(&diff2, delta2);
        prevClusterPos[tid] = clusterPos[tid];  // Save clusterPos
      }
      __syncthreads();

      if (tid == 0) {
        float diff = sqrtf(diff2);
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
    } while (notDone);
    if (tid == 0)
      pfcIter[topoId] = iter;
  }

  // For clustering largest topos
  __device__ void dev_hcalFastCluster_original(int topoId,
                                               int nSeeds,
                                               int nRHTopo,
                                               const float* __restrict__ pfrh_x,
                                               const float* __restrict__ pfrh_y,
                                               const float* __restrict__ pfrh_z,
                                               const float* __restrict__ pfrh_energy,
                                               const int* __restrict__ pfrh_layer,
                                               const int* __restrict__ pfrh_depth,
                                               const int* __restrict__ pfrh_neighbours,
                                               float* pcrhfrac,
                                               int* pcrhfracind,
                                               int* seedFracOffsets,
                                               int* topoSeedOffsets,
                                               int* topoSeedList,
                                               int* pfcIter) {
    __shared__ int nRHNotSeed, topoSeedBegin, gridStride, iter;
    __shared__ float tol, diff2, rhENormInv;
    __shared__ bool notDone, debug;
    __shared__ float4 clusterPos[400], prevClusterPos[400];
    __shared__ float clusterEnergy[400], rhFracSum[1500];
    __shared__ int seeds[400], rechits[1500];

    if (threadIdx.x == 0) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = topoSeedOffsets[topoId];
      tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
      gridStride = blockDim.x;
      iter = 0;
      notDone = true;
      //debug = true;
      debug = false;
      //int s0 = topoSeedList[topoSeedBegin];
      //debug = (s0==33 || s0==49 || s0==50 || s0==57 || s0==58 || s0==59 || s0==36 || s0==37 || s0==39 || s0==46 || s0==55 || s0==61 || s0==63 || s0==99 || s0==105 || s0==112 || s0==114 || s0==122 || s0==109 || s0==116 || s0==66 || s0==67 || s0==72 || s0==74 || s0==75 || s0==81 || s0==70 || s0==76 || s0==78 || s0==353 || s0==360 || s0==362 || s0==368 || s0==377 || s0==378 || s0==366 || s0==373 || s0==375 || s0==382 || s0==2 || s0==11 || s0==16 || s0==26 || s0==12 || s0==14 || s0==22 || s0==29 || s0==31 || s0==323 || s0==328 || s0==331 || s0==344 || s0==345 || s0==325 || s0==326 || s0==349 || s0==163 || s0==177 || s0==165 || s0==175 || s0==182 || s0==190 || s0==130 || s0==138 || s0==146 || s0==153 || s0==155 || s0==135 || s0==141 || s0==149 || s0==151 || s0==157 || s0==159 || s0==234 || s0==248 || s0==250 || s0==254 || s0==418 || s0==424 || s0==421 || s0==422 || s0==429 || s0==431 || s0==436 || s0==438 || s0==444 || s0==446 || s0==202 || s0==209 || s0==196 || s0==207 || s0==384 || s0==386 || s0==394 || s0==402 || s0==411 || s0==388 || s0==391 || s0==397 || s0==405 || s0==412 || s0==414 || s0==305 || s0==309 || s0==480 || s0==483 || s0==488 || s0==498 || s0==506 || s0==486 || s0==494 || s0==500 || s0==509 || s0==259 || s0==264 || s0==267 || s0==275 || s0==280 || s0==282 || s0==269 || s0==271 || s0==278 || s0==284 || s0==450 || s0==458 || s0==466 || s0==472 || s0==475 || s0==469 || s0==479 || s0==548 || s0==550 || s0==565 || s0==608 || s0==617 || s0==613 || s0==615 || s0==620 || s0==623 || s0==639 || s0==592 || s0==595 || s0==589 || s0==597 || s0==599 || s0==604 || s0==866 || s0==872 || s0==874 || s0==881 || s0==886 || s0==892 || s0==841 || s0==858 || s0==846 || s0==855 || s0==860 || s0==701 || s0==702 || s0==643 || s0==646 || s0==654 || s0==662 || s0==739 || s0==746 || s0==752 || s0==755 || s0==740 || s0==743 || s0==759 || s0==767 || s0==531 || s0==525 || s0==527 || s0==543 || s0==929 || s0==945 || s0==934 || s0==956 || s0==959 || s0==704 || s0==705 || s0==707 || s0==714 || s0==720 || s0==723 || s0==728 || s0==729 || s0==730 || s0==731 || s0==708 || s0==710 || s0==718 || s0==725 || s0==732 || s0==735 || s0==993 || s0==995 || s0==1008 || s0==1010 || s0==1014 || s0==1015 || s0==1020 || s0==896 || s0==898 || s0==920 || s0==922 || s0==903 || s0==910 || s0==918 || s0==800 || s0==809 || s0==804 || s0==805 || s0==822 || s0==777 || s0==779 || s0==786 || s0==792 || s0==794 || s0==772 || s0==780 || s0==781 || s0==798 || s0==960 || s0==978 || s0==984 || s0==964 || s0==966 || s0==973 || s0==975 || s0==980 || s0==981 || s0==1059 || s0==1072 || s0==1081 || s0==1070 || s0==1076 || s0==1079 || s0==1086 || s0==1147 || s0==1124 || s0==1135 || s0==1141 || s0==1149 || s0==1090 || s0==1098 || s0==1114 || s0==1095 || s0==1109 || s0==1119 || s0==1386 || s0==1387 || s0==1401 || s0==1396 || s0==1404 || s0==1347 || s0==1352 || s0==1355 || s0==1360 || s0==1185 || s0==1186 || s0==1194 || s0==1203 || s0==1210 || s0==1189 || s0==1207 || s0==1034 || s0==1040 || s0==1042 || s0==1049 || s0==1031 || s0==1044 || s0==1047 || s0==1154 || s0==1161 || s0==1177 || s0==1157 || s0==1164 || s0==1166 || s0==1172 || s0==1180 || s0==1249 || s0==1250 || s0==1251 || s0==1258 || s0==1266 || s0==1255 || s0==1260 || s0==1262 || s0==1270 || s0==1276 || s0==1409 || s0==1411 || s0==1418 || s0==1426 || s0==1427 || s0==1435 || s0==1415 || s0==1420 || s0==1422 || s0==1430 || s0==1441 || s0==1442 || s0==1443 || s0==1449 || s0==1458 || s0==1464 || s0==1467 || s0==1447 || s0==1461 || s0==1504 || s0==1515 || s0==1520 || s0==1529 || s0==1509 || s0==1511 || s0==1517 || s0==1524 || s0==1527 || s0==1532 || s0==1534 || s0==1320 || s0==1323 || s0==1331 || s0==1338 || s0==1317 || s0==1326 || s0==1335 || s0==1343 || s0==1218 || s0==1233 || s0==1234 || s0==1242 || s0==1243 || s0==1228 || s0==1230 || s0==1237 || s0==1239 || s0==1244 || s0==1481 || s0==1479 || s0==1492 || s0==1283 || s0==1289 || s0==1290 || s0==1304 || s0==1307 || s0==1293 || s0==1536 || s0==1539 || s0==1546 || s0==1553 || s0==1541 || s0==1557 || s0==1558 || s0==2114) ? true : false;

      //debug = (nSeeds == 2 && ( (topoSeedList[topoSeedBegin]==11 && topoSeedList[topoSeedBegin+1]==5) || (topoSeedList[topoSeedBegin]==5 && topoSeedList[topoSeedBegin+1]==11) )) ? true : false;

      int i = topoSeedList[topoSeedBegin];
      if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
        rhENormInv = recHitEnergyNormInvEB_vec[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = recHitEnergyNormInvEE_vec[pfrh_depth[i] - 1];
      else
        printf("Rechit %d has invalid layer %d!\n", i, pfrh_layer[i]);
    }
    __syncthreads();

    for (int n = threadIdx.x; n < nRHTopo; n += gridStride) {
      if (n < nSeeds)
        seeds[n] = topoSeedList[topoSeedBegin + n];
      if (n < nRHNotSeed - 1)
        rechits[n] = pcrhfracind[seedFracOffsets[topoSeedList[topoSeedBegin]] + n + 1];
    }
    __syncthreads();

    auto getSeedRhIdx = [&](int seedNum) { return seeds[seedNum]; };

    auto getRhFracIdx = [&](int seedNum, int rhNum) {
      if (rhNum <= 0)
        printf("Invalid rhNum (%d) for getRhFracIdx!\n", rhNum);
      return rechits[rhNum - 1];
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

    auto computeClusterPos = [&](float4& pos4, float frac, int rhInd, bool isDebug) {
      float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
      const auto rh_energy = pfrh_energy[rhInd] * frac;
      const auto norm = (frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
      if (isDebug)
        printf("\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
               rhInd,
               norm,
               frac,
               rh_energy,
               rechitPos.x,
               rechitPos.y,
               rechitPos.z);

      pos4.x += rechitPos.x * norm;
      pos4.y += rechitPos.y * norm;
      pos4.z += rechitPos.z * norm;
      pos4.w += norm;  //  position_norm
    };

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
      int i = seeds[s];
      clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
      prevClusterPos[s] = clusterPos[s];
      clusterEnergy[s] = pfrh_energy[i];
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        pcrhfracind[seedFracOffsets[i] + r + 1] = rechits[r];
        pcrhfrac[seedFracOffsets[i] + r + 1] = -1.;
      }
    }
    __syncthreads();

    do {
      if (debug && threadIdx.x == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      if (threadIdx.x == 0)
        diff2 = -1;
      // Reset rhFracSum
      for (int tid = threadIdx.x; tid < nRHNotSeed - 1; tid += gridStride) {
        rhFracSum[tid] = 0.;
        int rhThreadIdx = rechits[tid];
        float4 rhThreadPos = make_float4(pfrh_x[rhThreadIdx], pfrh_y[rhThreadIdx], pfrh_z[rhThreadIdx], 1.);
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / showerSigma2;
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          rhFracSum[tid] += fraction;
        }
      }
      __syncthreads();

      for (int tid = threadIdx.x; tid < nRHNotSeed - 1; tid += gridStride) {
        int rhThreadIdx = rechits[tid];
        float4 rhThreadPos = make_float4(pfrh_x[rhThreadIdx], pfrh_y[rhThreadIdx], pfrh_z[rhThreadIdx], 1.);
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / showerSigma2;
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > minFracTot) {
            float fracpct = fraction / rhFracSum[tid];
            //float fracpct = pcrhfrac[seedFracOffsets[i]+tid+1] / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
              pcrhfrac[seedFracOffsets[i] + tid + 1] = fracpct;
            } else {
              pcrhfrac[seedFracOffsets[i] + tid + 1] = -1;
            }
          } else {
            pcrhfrac[seedFracOffsets[i] + tid + 1] = -1;
          }
        }
      }
      __syncthreads();

      if (debug && threadIdx.x == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      // Reset cluster position and energy
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int seedRhIdx = getSeedRhIdx(s);
        float norm = logf(pfrh_energy[seedRhIdx] * rhENormInv);
        clusterPos[s] = make_float4(pfrh_x[seedRhIdx] * norm, pfrh_y[seedRhIdx] * norm, pfrh_z[seedRhIdx] * norm, norm);
        clusterEnergy[s] = pfrh_energy[seedRhIdx];
        if (debug) {
          printf("Cluster %d (seed %d) has energy %f\tpos = (%f, %f, %f, %f)\n",
                 s,
                 seeds[s],
                 clusterEnergy[s],
                 clusterPos[s].x,
                 clusterPos[s].y,
                 clusterPos[s].z,
                 clusterPos[s].w);
        }
      }
      __syncthreads();

      // Recalculate position
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int seedRhIdx = getSeedRhIdx(s);
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = getRhFrac(s, r + 1);

          if (frac > -0.5) {
            clusterEnergy[s] += frac * pfrh_energy[j];

            if (nSeeds == 1 || j == pfrh_neighbours[8 * seedRhIdx] || j == pfrh_neighbours[8 * seedRhIdx + 1] ||
                j == pfrh_neighbours[8 * seedRhIdx + 2] || j == pfrh_neighbours[8 * seedRhIdx + 3])
              computeClusterPos(clusterPos[s], frac, j, debug);
          }
        }
      }
      __syncthreads();

      // Position normalization
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        if (clusterPos[s].w >= minAllowedNormalization) {
          // Divide by position norm
          clusterPos[s].x /= clusterPos[s].w;
          clusterPos[s].y /= clusterPos[s].w;
          clusterPos[s].z /= clusterPos[s].w;

          if (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   s,
                   seeds[s],
                   clusterEnergy[s],
                   clusterPos[s].x,
                   clusterPos[s].y,
                   clusterPos[s].z);
        } else {
          if (debug)
            printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                   s,
                   seeds[s],
                   clusterPos[s].w,
                   minAllowedNormalization);
          clusterPos[s].x = 0.0;
          clusterPos[s].y = 0.0;
          clusterPos[s].z = 0.0;
        }
      }
      __syncthreads();

      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
        if (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
        atomicMaxF(&diff2, delta2);
        prevClusterPos[s] = clusterPos[s];  // Save clusterPos
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        float diff = sqrtf(diff2);
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
    } while (notDone);
    if (threadIdx.x == 0)
      pfcIter[topoId] = iter;
  }

  __global__ void hcalFastCluster_optimizedSimple(size_t nRH,
                                                  const float* __restrict__ pfrh_x,
                                                  const float* __restrict__ pfrh_y,
                                                  const float* __restrict__ pfrh_z,
                                                  const float* __restrict__ pfrh_energy,
                                                  int* pfrh_topoId,
                                                  int* pfrh_isSeed,
                                                  const int* __restrict__ pfrh_layer,
                                                  const int* __restrict__ pfrh_depth,
                                                  float* pcrhfrac,
                                                  int* pcrhfracind,
                                                  int* topoSeedCount,
                                                  int* topoRHCount,
                                                  int* seedFracOffsets,
                                                  int* topoSeedOffsets,
                                                  int* topoSeedList,
                                                  int* pfcIter) {
    int topoId = blockIdx.x;
    int r = threadIdx.x;  // thread index is rechit number
    // Only cluster single seed topos
    if (topoId < nRH && topoRHCount[topoId] > 1 && topoRHCount[topoId] < 33 && topoSeedCount[topoId] == 1 &&
        topoRHCount[topoId] != topoSeedCount[topoId]) {
      __shared__ int i, iter;
      __shared__ float tol, diff, diff2, clusterEnergy;
      __shared__ float4 clusterPos, prevClusterPos;
      __shared__ bool notDone, debug;
      if (r == 0) {
        i = topoSeedList[topoSeedOffsets[topoId]];  // i is the seed rechit index
        clusterPos = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.);
        prevClusterPos = clusterPos;
        clusterEnergy = pfrh_energy[i];
        tol = stoppingTolerance;  // stopping tolerance * tolerance scaling
        iter = 0;
        notDone = true;
        debug = false;
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
      }
      __syncthreads();
      int j = -1;
      // Populate rechit index array
      if (r < topoRHCount[topoId])
        j = pcrhfracind[seedFracOffsets[i] + r];  // rechit index for this thread
      int rhFracOffset = seedFracOffsets[i] + r;  // Offset for this rechit in pcrhfrac, pcrhfracidx arrays
      float4 rhPos;
      float rhENormInv = -1., rhEnergy = -1., rhPosNorm = -1.;
      if (j > -1) {
        rhPos = make_float4(pfrh_x[j], pfrh_y[j], pfrh_z[j], 1.);
        rhEnergy = pfrh_energy[j];

        if (pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
          rhENormInv = recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1];
        } else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) {
          rhENormInv = recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1];
        } else {
          printf("Rechit %d has invalid layer %d!\n", j, pfrh_layer[j]);
        }

        rhPosNorm = fmaxf(0., logf(rhEnergy * rhENormInv));
      }
      __syncthreads();

      do {
        if (debug && r == 0) {
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }
        float dist2 = -1., d2 = -1., fraction = -1.;
        if (j > -1) {
          dist2 = (clusterPos.x - rhPos.x) * (clusterPos.x - rhPos.x) +
                  (clusterPos.y - rhPos.y) * (clusterPos.y - rhPos.y) +
                  (clusterPos.z - rhPos.z) * (clusterPos.z - rhPos.z);

          d2 = dist2 / showerSigma2;
          fraction = clusterEnergy * rhENormInv * expf(-0.5 * d2);

          // For single seed clusters, rechit fraction is either 1 (100%) or -1 (not included)
          if (fraction > minFracTot && d2 < 100.)
            fraction = 1.;
          else
            fraction = -1.;
          pcrhfrac[rhFracOffset] = fraction;
        }
        __syncthreads();

        if (debug && r == 0)
          printf("Computing cluster position for topoId %d\n", topoId);

        if (r == 0) {
          // Reset cluster position and energy
          clusterPos = make_float4(0.0, 0.0, 0.0, 0.0);
          clusterEnergy = 0;
        }
        __syncthreads();

        // Recalculate cluster position and energy
        if (fraction > -0.5) {
          atomicAdd(&clusterEnergy, rhEnergy);
          //computeClusterPos(clusterPos, rechitPos, rhEnergy, rhENormInv, debug);
          atomicAdd(&clusterPos.x, rhPos.x * rhPosNorm);
          atomicAdd(&clusterPos.y, rhPos.y * rhPosNorm);
          atomicAdd(&clusterPos.z, rhPos.z * rhPosNorm);
          atomicAdd(&clusterPos.w, rhPosNorm);  // position_norm
        }
        __syncthreads();

        if (r == 0) {
          // Normalize the seed postiion
          if (clusterPos.w >= minAllowedNormalization) {
            // Divide by position norm
            clusterPos.x /= clusterPos.w;
            clusterPos.y /= clusterPos.w;
            clusterPos.z /= clusterPos.w;

            if (debug)
              printf("\tPF cluster (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                     i,
                     clusterEnergy,
                     clusterPos.x,
                     clusterPos.y,
                     clusterPos.z);
          } else {
            if (debug)
              printf("\tPF cluster (seed %d) position norm (%f) less than minimum (%f)\n",
                     i,
                     clusterPos.w,
                     minAllowedNormalization);
            clusterPos.x = 0.;
            clusterPos.y = 0.;
            clusterPos.z = 0.;
          }
          diff2 = dR2(prevClusterPos, clusterPos);
          if (debug)
            printf("\tPF cluster (seed %d) has diff2 = %f\n", i, diff2);
          prevClusterPos = clusterPos;  // Save clusterPos

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
      } while (notDone);
      if (r == 0)
        pfcIter[topoId] = iter;
    } else if (r == 0 && (topoRHCount[topoId] == 1 ||
                          (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
      // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
      pfcIter[topoId] = 0;
    }
  }

  __global__ void hcalFastCluster_optimizedComplex(size_t nRH,
                                                   const float* __restrict__ pfrh_x,
                                                   const float* __restrict__ pfrh_y,
                                                   const float* __restrict__ pfrh_z,
                                                   const float* __restrict__ pfrh_energy,
                                                   int* pfrh_topoId,
                                                   int* pfrh_isSeed,
                                                   const int* __restrict__ pfrh_layer,
                                                   const int* __restrict__ pfrh_depth,
                                                   const int* __restrict__ neigh4_Ind,
                                                   float* pcrhfrac,
                                                   int* pcrhfracind,
                                                   float* fracSum,
                                                   int* rhCount,
                                                   int* topoSeedCount,
                                                   int* topoRHCount,
                                                   int* seedFracOffsets,
                                                   int* topoSeedOffsets,
                                                   int* topoSeedList,
                                                   float4* _clusterPos,
                                                   float4* _prevClusterPos,
                                                   float* _clusterEnergy,
                                                   int* pfcIter) {
    int topoId = blockIdx.x;
    int tid = threadIdx.x;

    // Exclude topo clusters with >= 3 seeds for testing
    //if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId] && (blockDim.x <= 32 ? (topoSeedCount[topoId] < 3) : (topoSeedCount[topoId] >= 3) ))  {

    //if (topoId < nRH && topoRHCount[topoId] > 1 && topoRHCount[topoId] < 33 && topoSeedCount[topoId] == 1 && topoRHCount[topoId] != topoSeedCount[topoId])
    if (topoId < nRH && topoRHCount[topoId] > 1 && (topoRHCount[topoId] - topoSeedCount[topoId]) < 256 &&
        topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId]) {
      //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
      __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
      __shared__ float tol, diff2, rhENormInv;
      __shared__ bool notDone, debug;
      __shared__ float4 clusterPos[100], prevClusterPos[100];
      __shared__ float clusterEnergy[100], rhFracSum[256];
      __shared__ int seeds[100], rechits[256];

      if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
        //gridStride = blockDim.x * gridDim.x;
        gridStride = blockDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        //debug = true;
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
        //debug = (topoId == 1 || topoId == 5 || topoId == 6 || topoId == 8 || topoId == 9 || topoId == 10 || topoId == 12 || topoId == 13) ? true : false;

        int i = topoSeedList[topoSeedBegin];
        if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
          rhENormInv = recHitEnergyNormInvEB_vec[pfrh_depth[i] - 1];
        else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
          rhENormInv = recHitEnergyNormInvEE_vec[pfrh_depth[i] - 1];
        else
          printf("Rechit %d has invalid layer %d!\n", i, pfrh_layer[i]);
      }
      __syncthreads();

      for (int n = threadIdx.x; n < nRHTopo; n += gridStride) {
        if (n < nSeeds)
          seeds[n] = topoSeedList[topoSeedBegin + n];
        if (n < nRHNotSeed - 1)
          rechits[n] = pcrhfracind[seedFracOffsets[topoSeedList[topoSeedBegin]] + n + 1];
      }
      __syncthreads();

      auto getSeedRhIdx = [&](int seedNum) { return seeds[seedNum]; };

      auto getRhFracIdx = [&](int seedNum, int rhNum) {
        if (rhNum <= 0)
          printf("Invalid rhNum (%d) for getRhFracIdx!\n", rhNum);
        return rechits[rhNum - 1];
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

      auto computeClusterPos = [&](float4& pos4, float frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        const auto rh_energy = pfrh_energy[rhInd] * frac;
        const auto norm = (frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
        if (isDebug)
          printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
                 rhInd,
                 norm,
                 frac,
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
    auto computeClusterPosAtomic = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm =
            (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
        if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);

        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);   // position_norm
    };
*/

      // Set initial cluster position (energy) to seed rechit position (energy)
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[s] = clusterPos[s];
        clusterEnergy[s] = pfrh_energy[i];
        for (int r = 0; r < (nRHNotSeed - 1); r++) {
          pcrhfracind[seedFracOffsets[i] + r + 1] = rechits[r];
          pcrhfrac[seedFracOffsets[i] + r + 1] = -1.;
        }
      }
      __syncthreads();

      int rhThreadIdx = -1;
      float4 rhThreadPos;
      if (tid < (nRHNotSeed - 1)) {
        rhThreadIdx = rechits[tid];  // Index when thread represents rechit
        rhThreadPos = make_float4(pfrh_x[rhThreadIdx], pfrh_y[rhThreadIdx], pfrh_z[rhThreadIdx], 1.);
      }

      // Neighbors when threadIdx represents seed
      int seedThreadIdx = -1;
      int4 seedNeighbors = make_int4(-9, -9, -9, -9);
      float seedEnergy = -1.;
      float4 seedInitClusterPos = make_float4(0., 0., 0., 0.);
      if (tid < nSeeds) {
        seedThreadIdx = getSeedRhIdx(tid);
        seedNeighbors = make_int4(neigh4_Ind[nNeigh * seedThreadIdx],
                                  neigh4_Ind[nNeigh * seedThreadIdx + 1],
                                  neigh4_Ind[nNeigh * seedThreadIdx + 2],
                                  neigh4_Ind[nNeigh * seedThreadIdx + 3]);
        seedEnergy = pfrh_energy[seedThreadIdx];

        // Compute initial cluster position shift for seed
        computeClusterPos(seedInitClusterPos, 1., seedThreadIdx, debug);
      }

      do {
        if (debug && threadIdx.x == 0) {
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }

        // Reset fracSum
        rhFracSum[tid] = 0.;
        if (tid == 0)
          diff2 = -1;

        if (tid < (nRHNotSeed - 1)) {
          for (int s = 0; s < nSeeds; s++) {
            float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                          (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                          (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

            float d2 = dist2 / showerSigma2;
            float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);
            //pcrhfrac[seedFracOffsets[seeds[s]]+tid+1] = fraction;

            rhFracSum[tid] += fraction;
          }
        }
        __syncthreads();

        if (tid < (nRHNotSeed - 1)) {
          for (int s = 0; s < nSeeds; s++) {
            int i = seeds[s];
            float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                          (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                          (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

            float d2 = dist2 / showerSigma2;
            float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

            if (rhFracSum[tid] > minFracTot) {
              float fracpct = fraction / rhFracSum[tid];
              //float fracpct = pcrhfrac[seedFracOffsets[i]+tid+1] / rhFracSum[tid];
              if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                pcrhfrac[seedFracOffsets[i] + tid + 1] = fracpct;
              } else {
                pcrhfrac[seedFracOffsets[i] + tid + 1] = -1;
              }
            } else {
              pcrhfrac[seedFracOffsets[i] + tid + 1] = -1;
            }
          }
        }
        __syncthreads();

        if (debug && tid == 0)
          printf("Computing cluster position for topoId %d\n", topoId);

        // Reset cluster position and energy
        if (tid < nSeeds) {
          clusterPos[tid] = seedInitClusterPos;
          clusterEnergy[tid] = seedEnergy;
          if (debug) {
            printf("Cluster %d (seed %d) has energy %f\tpos = (%f, %f, %f, %f)\n",
                   tid,
                   seeds[tid],
                   clusterEnergy[tid],
                   clusterPos[tid].x,
                   clusterPos[tid].y,
                   clusterPos[tid].z,
                   clusterPos[tid].w);
          }
        }

        __syncthreads();

        // Recalculate position
        if (tid < nSeeds) {
          for (int r = 0; r < nRHNotSeed - 1; r++) {
            int j = rechits[r];
            float frac = getRhFrac(tid, r + 1);

            if (frac > -0.5) {
              clusterEnergy[tid] += frac * pfrh_energy[j];

              if (nSeeds == 1 || j == seedNeighbors.x || j == seedNeighbors.y || j == seedNeighbors.z ||
                  j == seedNeighbors.w)
                computeClusterPos(clusterPos[tid], frac, j, debug);
            }
          }
        }
        __syncthreads();

        // Position normalization
        if (tid < nSeeds) {
          if (clusterPos[tid].w >= minAllowedNormalization) {
            // Divide by position norm
            clusterPos[tid].x /= clusterPos[tid].w;
            clusterPos[tid].y /= clusterPos[tid].w;
            clusterPos[tid].z /= clusterPos[tid].w;

            if (debug)
              printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                     tid,
                     seedThreadIdx,
                     clusterEnergy[tid],
                     clusterPos[tid].x,
                     clusterPos[tid].y,
                     clusterPos[tid].z);
          } else {
            if (debug)
              printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                     tid,
                     seedThreadIdx,
                     clusterPos[tid].w,
                     minAllowedNormalization);
            clusterPos[tid].x = 0.0;
            clusterPos[tid].y = 0.0;
            clusterPos[tid].z = 0.0;
            //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
          }
        }
        __syncthreads();

        if (tid < nSeeds) {
          float delta2 = dR2(prevClusterPos[tid], clusterPos[tid]);
          if (debug)
            printf("\tCluster %d (seed %d) has delta2 = %f\n", tid, seeds[tid], delta2);
          atomicMaxF(&diff2, delta2);
          //            if (delta2 > diff2) {
          //                diff2 = delta2;
          //                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
          //            }

          prevClusterPos[tid] = clusterPos[tid];  // Save clusterPos
        }
        __syncthreads();

        if (tid == 0) {
          float diff = sqrtf(diff2);
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
      } while (notDone);
      if (tid == 0)
        pfcIter[topoId] = iter;
    } else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 ||
                                    (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
      // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
      pfcIter[topoId] = 0;
    }
  }

  __global__ void hcalFastCluster_sharedRHList(size_t nRH,
                                               const float* __restrict__ pfrh_x,
                                               const float* __restrict__ pfrh_y,
                                               const float* __restrict__ pfrh_z,
                                               const float* __restrict__ pfrh_energy,
                                               int* pfrh_topoId,
                                               int* pfrh_isSeed,
                                               const int* __restrict__ pfrh_layer,
                                               const int* __restrict__ pfrh_depth,
                                               const int* __restrict__ neigh4_Ind,
                                               float* pcrhfrac,
                                               int* pcrhfracind,
                                               float* fracSum,
                                               int* rhCount,
                                               int* topoSeedCount,
                                               int* topoRHCount,
                                               int* seedFracOffsets,
                                               int* topoSeedOffsets,
                                               int* topoSeedList,
                                               float4* _clusterPos,
                                               float4* _prevClusterPos,
                                               float* _clusterEnergy,
                                               int* pfcIter) {
    int topoId = blockIdx.x;

    // Exclude topo clusters with >= 3 seeds for testing
    //if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId] && (blockDim.x <= 32 ? (topoSeedCount[topoId] < 3) : (topoSeedCount[topoId] >= 3) ))  {

    //if (topoId < nRH && topoRHCount[topoId] > 1 && topoRHCount[topoId] < 33 && topoSeedCount[topoId] == 1 && topoRHCount[topoId] != topoSeedCount[topoId])
    if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 &&
        topoRHCount[topoId] != topoSeedCount[topoId]) {
      //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
      __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
      __shared__ float tol, diff, diff2, rhENormInv;
      __shared__ bool notDone, debug, noPosCalc;
      __shared__ float4 clusterPos[100], prevClusterPos[100];
      __shared__ float clusterEnergy[100];
      __shared__ int seeds[100], rechits[256];

      if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
        //gridStride = blockDim.x * gridDim.x;
        gridStride = blockDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        //debug = true;
        noPosCalc = false;
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
        //debug = (topoId == 1 || topoId == 5 || topoId == 6 || topoId == 8 || topoId == 9 || topoId == 10 || topoId == 12 || topoId == 13) ? true : false;

        int i = topoSeedList[topoSeedBegin];
        if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
          rhENormInv = recHitEnergyNormInvEB_vec[pfrh_depth[i] - 1];
        else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
          rhENormInv = recHitEnergyNormInvEE_vec[pfrh_depth[i] - 1];
        else
          printf("Rechit %d has invalid layer %d!\n", i, pfrh_layer[i]);
      }
      __syncthreads();

      for (int n = threadIdx.x; n < nRHTopo; n += gridStride) {
        if (n < nSeeds)
          seeds[n] = topoSeedList[topoSeedBegin + n];
        if (n < nRHNotSeed - 1)
          rechits[n] = pcrhfracind[seedFracOffsets[topoSeedList[topoSeedBegin]] + n + 1];
      }
      __syncthreads();

      auto getSeedRhIdx = [&](int seedNum) { return seeds[seedNum]; };

      auto getRhFracIdx = [&](int seedNum, int rhNum) {
        if (rhNum <= 0)
          printf("Invalid rhNum (%d) for getRhFracIdx!\n", rhNum);
        return rechits[rhNum - 1];
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

      auto computeClusterPos = [&](float4& pos4, float frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        const auto rh_energy = pfrh_energy[rhInd] * frac;
        const auto norm = (frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
        if (isDebug)
          printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
                 rhInd,
                 norm,
                 frac,
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
    auto computeClusterPosAtomic = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm =
            (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
        if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);

        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);   // position_norm
    };
*/

      // Set initial cluster position (energy) to seed rechit position (energy)
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[s] = clusterPos[s];
        clusterEnergy[s] = pfrh_energy[i];
      }
      __syncthreads();

      do {
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
            // Ensure rh frac indicies have same order for all seeds
            pcrhfracind[seedFracOffsets[i] + r] = j;
          }
        }
        __syncthreads();

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                              // PF clusters
            int i = getSeedRhIdx(s);
            int j = getRhFracIdx(s, r);

            if (debug) {
              printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n",
                     s,
                     i,
                     clusterPos[s].x,
                     clusterPos[s].y,
                     clusterPos[s].z);
            }

            float dist2 = (clusterPos[s].x - pfrh_x[j]) * (clusterPos[s].x - pfrh_x[j]) +
                          (clusterPos[s].y - pfrh_y[j]) * (clusterPos[s].y - pfrh_y[j]) +
                          (clusterPos[s].z - pfrh_z[j]) * (clusterPos[s].z - pfrh_z[j]);

            float d2 = dist2 / showerSigma2;
            float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

            fracSum[j] += fraction;
            //atomicAdd(&fracSum[j],fraction);
            //                if( pfrh_isSeed[j]!=1) {
            //                    atomicAdd(&fracSum[j],fraction);
            //                }
          }
        }
        __syncthreads();

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                              // PF clusters
            int i = getSeedRhIdx(s);
            int j = getRhFracIdx(s, r);

            float dist2 = (clusterPos[s].x - pfrh_x[j]) * (clusterPos[s].x - pfrh_x[j]) +
                          (clusterPos[s].y - pfrh_y[j]) * (clusterPos[s].y - pfrh_y[j]) +
                          (clusterPos[s].z - pfrh_z[j]) * (clusterPos[s].z - pfrh_z[j]);

            float d2 = dist2 / showerSigma2;
            float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);
            //if(fraction < 0.) printf("FRACTION is NEGATIVE!!!");

            if (fracSum[j] > minFracTot) {
              float fracpct = fraction / fracSum[j];
              if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                pcrhfrac[seedFracOffsets[i] + r] = fracpct;
              } else {
                pcrhfrac[seedFracOffsets[i] + r] = -1;
              }
            } else {
              pcrhfrac[seedFracOffsets[i] + r] = -1;
            }
          }
        }
        __syncthreads();

        if (!noPosCalc) {
          if (debug && threadIdx.x == 0)
            printf("Computing cluster position for topoId %d\n", topoId);

          // Reset cluster position and energy
          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            clusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[s] = 0;
          }
          __syncthreads();

          // Recalculate position
          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {  // PF clusters
            int i = getSeedRhIdx(s);                                // Seed index
            // Seed rechit first
            clusterEnergy[s] += pfrh_energy[i];
            computeClusterPos(clusterPos[s], 1., i, debug);
            for (int r = 1; r < nRHNotSeed; r++) {  // Rechits
              if (debug) {
                printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                for (int k = 0; k < nNeigh; k++) {
                  if (k != 0)
                    printf(", ");
                  printf("%d", neigh4_Ind[nNeigh * i + k]);
                }
                printf("]\n");
              }

              // Calculate cluster energy by summing rechit fractional energies
              int j = getRhFracIdx(s, r);
              float frac = getRhFrac(s, r);

              if (frac > -0.5) {
                //if (debug)
                //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                clusterEnergy[s] += frac * pfrh_energy[j];

                bool updateClusterPos = false;
                if (nSeeds == 1) {
                  if (debug)
                    printf("\t\tThis topo cluster has a single seed.\n");
                  updateClusterPos = true;
                } else {
                  if (j == i) {
                    // This is the seed
                    updateClusterPos = true;
                  } else {
                    // Check if this is one of the neighboring rechits
                    for (int k = 0; k < nNeigh; k++) {
                      if (neigh4_Ind[nNeigh * i + k] < 0)
                        continue;
                      if (neigh4_Ind[nNeigh * i + k] == j) {
                        // Found it
                        if (debug)
                          printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                        updateClusterPos = true;
                        break;
                      }
                    }
                  }
                }
                if (updateClusterPos)
                  computeClusterPos(clusterPos[s], frac, j, debug);
              }
            }  // rechit loop
          }    // seed loop
          __syncthreads();

          // Normalize the seed postiions
          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            //int i = getSeedRhIdx(s);    // Seed index
            if (clusterPos[s].w >= minAllowedNormalization) {
              // Divide by position norm
              clusterPos[s].x /= clusterPos[s].w;
              clusterPos[s].y /= clusterPos[s].w;
              clusterPos[s].z /= clusterPos[s].w;

              if (debug)
                printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                       s,
                       getSeedRhIdx(s),
                       clusterEnergy[s],
                       clusterPos[s].x,
                       clusterPos[s].y,
                       clusterPos[s].z);
            } else {
              if (debug)
                printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                       s,
                       getSeedRhIdx(s),
                       clusterPos[s].w,
                       minAllowedNormalization);
              clusterPos[s].x = 0.0;
              clusterPos[s].y = 0.0;
              clusterPos[s].z = 0.0;
              //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
            }
          }

          // Reset diff2
          if (threadIdx.x == 0) {
            diff2 = -1.;
          }
          __syncthreads();

          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
            if (debug)
              printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
            atomicMaxF(&diff2, delta2);
            //            if (delta2 > diff2) {
            //                diff2 = delta2;
            //                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
            //            }

            prevClusterPos[s] = clusterPos[s];  // Save clusterPos
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
        }  // if (!noPosCalc)
        else {
          if (threadIdx.x == 0)
            notDone = false;
          __syncthreads();
        }
      } while (notDone);
      if (threadIdx.x == 0)
        pfcIter[topoId] = iter;
    } else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 ||
                                    (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
      // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
      pfcIter[topoId] = 0;
    }
  }

  __global__ void hcalFastCluster_original(size_t nRH,
                                           const float* __restrict__ pfrh_x,
                                           const float* __restrict__ pfrh_y,
                                           const float* __restrict__ pfrh_z,
                                           const float* __restrict__ pfrh_energy,
                                           int* pfrh_topoId,
                                           int* pfrh_isSeed,
                                           const int* __restrict__ pfrh_layer,
                                           const int* __restrict__ pfrh_depth,
                                           const int* __restrict__ neigh4_Ind,
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
                                           float* clusterEnergy,
                                           int* pfcIter) {
    int topoId = blockIdx.x;

    // Exclude topo clusters with >= 3 seeds for testing
    //if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId] && (blockDim.x <= 32 ? (topoSeedCount[topoId] < 3) : (topoSeedCount[topoId] >= 3) ))  {

    if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 &&
        topoRHCount[topoId] != topoSeedCount[topoId]) {
      //if (topoId < nRH && topoSeedCount[topoId] > 25 && topoSeedCount[topoId] > 24 && topoRHCount[topoId] != topoSeedCount[topoId]) {
      //if (topoId < nRH && topoRHCount[topoId] > topoSeedCount[topoId] && topoSeedCount[topoId] > 0 && topoSeedCount[topoId] < 25 && topoRHCount[topoId] != topoSeedCount[topoId]) {
      //if (topoId < nRH && topoRHCount[topoId] > topoSeedCount[topoId] && topoSeedCount[topoId] > 29 && topoRHCount[topoId] != topoSeedCount[topoId]) {
      //if (topoId < nRH && topoRHCount[topoId] > topoSeedCount[topoId] && topoSeedCount[topoId] > 0 && topoSeedCount[topoId] < 30 && topoRHCount[topoId] != topoSeedCount[topoId]) {
      //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
      __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
      __shared__ float tol, diff, diff2;
      __shared__ bool notDone, debug, noPosCalc;
      if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
        //gridStride = blockDim.x * gridDim.x;
        gridStride = blockDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        noPosCalc = false;
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
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

      auto computeClusterPos = [&](float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if (pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
          threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1];  // This number needs to be inverted
        } else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) {
          threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1];
        }

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
        atomicAdd(&pos4.w, norm);  // position_norm
                                   //        pos4.x += rechitPos.x * norm;
                                   //        pos4.y += rechitPos.y * norm;
                                   //        pos4.z += rechitPos.z * norm;
                                   //        pos4.w += norm;     //  position_norm
      };

      // Set initial cluster position (energy) to seed rechit position (energy)
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[i] = clusterPos[i];
        clusterEnergy[i] = pfrh_energy[i];
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

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
          for (int s = 0; s < nSeeds; s++) {                              // PF clusters
            int i = getSeedRhIdx(s);
            int j = getRhFracIdx(s, r);

            if (debug) {
              printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n",
                     s,
                     i,
                     clusterPos[i].x,
                     clusterPos[i].y,
                     clusterPos[i].z);
            }

            float dist2 = (clusterPos[i].x - pfrh_x[j]) * (clusterPos[i].x - pfrh_x[j]) +
                          (clusterPos[i].y - pfrh_y[j]) * (clusterPos[i].y - pfrh_y[j]) +
                          (clusterPos[i].z - pfrh_z[j]) * (clusterPos[i].z - pfrh_z[j]);

            float d2 = dist2 / showerSigma2;
            float fraction = -1.;

            if (pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
              fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
            } else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) {
              fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
            }
            if (fraction == -1.)
              printf("FRACTION is NEGATIVE!!!");

            if (pfrh_isSeed[j] != 1) {
              atomicAdd(&fracSum[j], fraction);
            }
          }
        }
        __syncthreads();

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

              if (pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
                fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
              } else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) {
                fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
              }
              if (fraction == -1.)
                printf("FRACTION is NEGATIVE!!!");

              if (fracSum[j] > minFracTot) {
                float fracpct = fraction / fracSum[j];
                if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                  pcrhfrac[seedFracOffsets[i] + r] = fracpct;
                } else {
                  pcrhfrac[seedFracOffsets[i] + r] = -1;
                }
              } else {
                pcrhfrac[seedFracOffsets[i] + r] = -1;
              }
            }
          }
        }
        __syncthreads();
        if (!noPosCalc) {
          if (debug && threadIdx.x == 0)
            printf("Computing cluster position for topoId %d\n", topoId);

          // Reset cluster position and energy
          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;
          }
          __syncthreads();

          // Recalculate position
          for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {  // One thread for each (non-seed) rechit
            for (int s = 0; s < nSeeds; s++) {                          // PF clusters
              int i = getSeedRhIdx(s);                                  // Seed index

              if (debug) {
                printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                for (int k = 0; k < nNeigh; k++) {
                  if (k != 0)
                    printf(", ");
                  printf("%d", neigh4_Ind[nNeigh * i + k]);
                }
                printf("]\n");
              }

              // Calculate cluster energy by summing rechit fractional energies
              int j = getRhFracIdx(s, r);
              float frac = getRhFrac(s, r);

              if (frac > -0.5) {
                //if (debug)
                //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                //clusterEnergy[i] += frac * pfrh_energy[j];
                atomicAdd(&clusterEnergy[i], frac * pfrh_energy[j]);

                bool updateClusterPos = false;
                if (nSeeds == 1) {
                  if (debug)
                    printf("\t\tThis topo cluster has a single seed.\n");
                  //computeClusterPos(clusterPos[i], frac, j, debug);
                  updateClusterPos = true;
                } else {
                  if (j == i) {
                    // This is the seed
                    //computeClusterPos(clusterPos[i], frac, j, debug);
                    updateClusterPos = true;
                  } else {
                    // Check if this is one of the neighboring rechits
                    for (int k = 0; k < nNeigh; k++) {
                      if (neigh4_Ind[nNeigh * i + k] < 0)
                        continue;
                      if (neigh4_Ind[nNeigh * i + k] == j) {
                        // Found it
                        if (debug)
                          printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                        //computeClusterPos(clusterPos[i], frac, j, debug);
                        updateClusterPos = true;
                        break;
                      }
                    }
                  }
                }
                if (updateClusterPos)
                  computeClusterPos(clusterPos[i], frac, j, debug);
              }
              //else if (debug)
              //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
            }
          }
          __syncthreads();

          // Normalize the seed postiions
          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);  // Seed index
            if (clusterPos[i].w >= minAllowedNormalization) {
              // Divide by position norm
              clusterPos[i].x /= clusterPos[i].w;
              clusterPos[i].y /= clusterPos[i].w;
              clusterPos[i].z /= clusterPos[i].w;

              if (debug)
                printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                       s,
                       i,
                       clusterEnergy[i],
                       clusterPos[i].x,
                       clusterPos[i].y,
                       clusterPos[i].z);
            } else {
              if (debug)
                printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                       s,
                       i,
                       clusterPos[i].w,
                       minAllowedNormalization);
              clusterPos[i].x = 0.0;
              clusterPos[i].y = 0.0;
              clusterPos[i].z = 0.0;
              //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
            }
          }

          // Reset diff2
          if (threadIdx.x == 0) {
            diff2 = -1.;
          }
          __syncthreads();

          for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug)
              printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
            atomicMaxF(&diff2, delta2);
            //            if (delta2 > diff2) {
            //                diff2 = delta2;
            //                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
            //            }

            prevClusterPos[i] = clusterPos[i];  // Save clusterPos
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
        }  // if (!noPosCalc)
        else {
          if (threadIdx.x == 0)
            notDone = false;
          __syncthreads();
        }
      }  // end while loop
      if (threadIdx.x == 0)
        pfcIter[topoId] = iter;
    } else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 ||
                                    (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
      // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
      pfcIter[topoId] = 0;
    }
  }

  __global__ void hcalFastCluster_selection(size_t nRH,
                                            const float* __restrict__ pfrh_x,
                                            const float* __restrict__ pfrh_y,
                                            const float* __restrict__ pfrh_z,
                                            const float* __restrict__ pfrh_energy,
                                            int* pfrh_topoId,
                                            int* pfrh_isSeed,
                                            const int* __restrict__ pfrh_layer,
                                            const int* __restrict__ pfrh_depth,
                                            const int* __restrict__ pfrh_neighbours,
                                            float* pcrhfrac,
                                            int* pcrhfracind,
                                            float* fracSum,
                                            int* rhCount,
                                            int* topoSeedCount,
                                            int* topoRHCount,
                                            int* seedFracOffsets,
                                            int* topoSeedOffsets,
                                            int* topoSeedList,
                                            float4* _clusterPos,
                                            float4* _prevClusterPos,
                                            float* _clusterEnergy,
                                            int* pfcIter) {
    __shared__ int topoId, nRHTopo, nSeeds;

    if (threadIdx.x == 0) {
      topoId = blockIdx.x;
      nRHTopo = topoRHCount[topoId];
      nSeeds = topoSeedCount[topoId];
    }
    __syncthreads();

    if (topoId < nRH && nRHTopo > 0 && nSeeds > 0) {
      if (nRHTopo == nSeeds) {
        // PF cluster is isolated seed. No iterations needed
        if (threadIdx.x == 0)
          pfcIter[topoId] = 0;
      } else if (nSeeds == 1) {
        // Single seed cluster
        dev_hcalFastCluster_optimizedSimple(topoId,
                                            nRHTopo,
                                            pfrh_x,
                                            pfrh_y,
                                            pfrh_z,
                                            pfrh_energy,
                                            pfrh_layer,
                                            pfrh_depth,
                                            pcrhfrac,
                                            pcrhfracind,
                                            topoSeedOffsets,
                                            topoSeedList,
                                            seedFracOffsets,
                                            pfcIter);
      } else if (nSeeds <= 100 && nRHTopo - nSeeds < 256) {
        dev_hcalFastCluster_optimizedComplex(topoId,
                                             nSeeds,
                                             nRHTopo,
                                             pfrh_x,
                                             pfrh_y,
                                             pfrh_z,
                                             pfrh_energy,
                                             pfrh_layer,
                                             pfrh_depth,
                                             pfrh_neighbours,
                                             pcrhfrac,
                                             pcrhfracind,
                                             seedFracOffsets,
                                             topoSeedOffsets,
                                             topoSeedList,
                                             pfcIter);
        //dev_hcalFastCluster_original(topoId, nSeeds, nRHTopo, pfrh_x, pfrh_y, pfrh_z, pfrh_energy, pfrh_layer, pfrh_depth, pfrh_neighbours, pcrhfrac, pcrhfracind, seedFracOffsets, topoSeedOffsets, topoSeedList, pfcIter);
      } else if (nSeeds <= 400 && (nRHTopo - nSeeds <= 1500)) {
        dev_hcalFastCluster_original(topoId,
                                     nSeeds,
                                     nRHTopo,
                                     pfrh_x,
                                     pfrh_y,
                                     pfrh_z,
                                     pfrh_energy,
                                     pfrh_layer,
                                     pfrh_depth,
                                     pfrh_neighbours,
                                     pcrhfrac,
                                     pcrhfracind,
                                     seedFracOffsets,
                                     topoSeedOffsets,
                                     topoSeedList,
                                     pfcIter);
      } else {
        if (threadIdx.x == 0)
          printf("ERROR: Topo cluster %d has %d seeds and %d rechits. SKIPPING!!\n", topoId, nSeeds, nRHTopo);
      }
    }
  }

  __global__ void hcalFastCluster_serialize(size_t nRH,
                                            const float* __restrict__ pfrh_x,
                                            const float* __restrict__ pfrh_y,
                                            const float* __restrict__ pfrh_z,
                                            const float* __restrict__ pfrh_energy,
                                            int* pfrh_topoId,
                                            int* pfrh_isSeed,
                                            const int* __restrict__ pfrh_layer,
                                            const int* __restrict__ pfrh_depth,
                                            const int* __restrict__ neigh4_Ind,
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
                                            float* clusterEnergy) {
    for (int topoId = 0; topoId < (int)nRH; topoId++) {
      int nSeeds = topoSeedCount[topoId];
      if (nSeeds < 1)
        continue;  // No seeds found for this topoId. Skip it
      int topoSeedBegin = topoSeedOffsets[topoId];

      int nRHTopo = topoRHCount[topoId];
      int iter = 0;

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

      //bool debug = true;
      bool debug = false;
      if (debug) {
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
          for (int r = 1; r < (nRHTopo - nSeeds + 1); r++) {
            if (r != 1)
              printf(", ");
            printf("%d", getRhFracIdx(0, r));
          }
          printf(")\n\n");
        }
      }

      float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // Tolerance scaling

      auto computeClusterPos = [&](float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if (pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
          threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1];  // This number needs to be inverted
        } else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) {
          threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1];
        }

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
          int i = getSeedRhIdx(s);

          if (iter == 0) {
            // Set initial cluster position to seed rechit position
            clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
            prevClusterPos[i] = clusterPos[i];

            // Set initial cluster energy to seed energy
            clusterEnergy[i] = pfrh_energy[i];
          } else {
            prevClusterPos[i] = clusterPos[i];

            if (debug) {
              printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n",
                     s,
                     i,
                     clusterPos[i].x,
                     clusterPos[i].y,
                     clusterPos[i].z);
            }

            // Reset cluster indices and fractions
            for (int n = (seedFracOffsets[i] + 1); n < (seedFracOffsets[i] + topoRHCount[topoId]); n++) {
              pcrhfrac[n] = -1.0;
            }
          }
          for (int r = 1; r < (nRHTopo - nSeeds + 1); r++) {
            int j = getRhFracIdx(s, r);

            float dist2 = (clusterPos[i].x - pfrh_x[j]) * (clusterPos[i].x - pfrh_x[j]) +
                          (clusterPos[i].y - pfrh_y[j]) * (clusterPos[i].y - pfrh_y[j]) +
                          (clusterPos[i].z - pfrh_z[j]) * (clusterPos[i].z - pfrh_z[j]);

            float d2 = dist2 / showerSigma2;
            float fraction = -1.;

            if (pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
              fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
            } else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) {
              fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
            }
            if (fraction == -1.)
              printf("FRACTION is NEGATIVE!!!");

            if (pfrh_isSeed[j] != 1) {
              atomicAdd(&fracSum[j], fraction);
            }
          }
        }
        for (int s = 0; s < nSeeds; s++) {  // PF clusters
          int i = getSeedRhIdx(s);

          for (int r = 1; r < (nRHTopo - nSeeds + 1); r++) {
            int j = getRhFracIdx(s, r);

            if (pfrh_isSeed[j] != 1) {
              float dist2 = (clusterPos[i].x - pfrh_x[j]) * (clusterPos[i].x - pfrh_x[j]) +
                            (clusterPos[i].y - pfrh_y[j]) * (clusterPos[i].y - pfrh_y[j]) +
                            (clusterPos[i].z - pfrh_z[j]) * (clusterPos[i].z - pfrh_z[j]);

              float d2 = dist2 / showerSigma2;
              float fraction = -1.;

              if (pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
                fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
              } else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) {
                fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
              }
              if (fraction == -1.)
                printf("FRACTION is NEGATIVE!!!");

              if (fracSum[j] > minFracTot) {
                float fracpct = fraction / fracSum[j];
                if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                  pcrhfrac[seedFracOffsets[i] + r] = fracpct;
                } else {
                  pcrhfrac[seedFracOffsets[i] + r] = -1;
                }
              } else {
                pcrhfrac[seedFracOffsets[i] + r] = -1;
              }
            }
          }
        }

        if (debug)
          printf("Computing cluster position for topoId %d\n", topoId);
        // Recalculate position
        for (int s = 0; s < nSeeds; s++) {  // PF clusters
          int i = getSeedRhIdx(s);          // Seed index

          if (debug) {
            printf("\tNow on seed %d\t\tneigh4Ind = [", i);
            for (int k = 0; k < nNeigh; k++) {
              if (k != 0)
                printf(", ");
              printf("%d", neigh4_Ind[nNeigh * i + k]);
            }
            printf("]\n");
          }
          // Zero out cluster position and energy
          clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
          clusterEnergy[i] = 0;

          // Calculate cluster energy by summing rechit fractional energies
          for (int r = 0; r < (nRHTopo - nSeeds + 1); r++) {
            int j = getRhFracIdx(s, r);
            float frac = getRhFrac(s, r);

            if (frac > -0.5) {
              //if (debug)
              //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
              clusterEnergy[i] += frac * pfrh_energy[j];

              if (nSeeds == 1) {
                if (debug)
                  printf("\t\tThis topo cluster has a single seed.\n");
                computeClusterPos(clusterPos[i], frac, j, debug);
              } else {
                if (j == i) {
                  // This is the seed
                  computeClusterPos(clusterPos[i], frac, j, debug);
                } else {
                  // Check if this is one of the neighboring rechits
                  for (int k = 0; k < nNeigh; k++) {
                    if (neigh4_Ind[nNeigh * i + k] < 0)
                      continue;
                    if (neigh4_Ind[nNeigh * i + k] == j) {
                      // Found it
                      if (debug)
                        printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                      computeClusterPos(clusterPos[i], frac, j, debug);
                    }
                  }
                }
              }
            }
            //else if (debug)
            //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
          }
          if (clusterPos[i].w >= minAllowedNormalization) {
            // Divide by position norm
            clusterPos[i].x /= clusterPos[i].w;
            clusterPos[i].y /= clusterPos[i].w;
            clusterPos[i].z /= clusterPos[i].w;

            if (debug)
              printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                     s,
                     i,
                     clusterEnergy[i],
                     clusterPos[i].x,
                     clusterPos[i].y,
                     clusterPos[i].z);
          } else {
            if (debug)
              printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                     s,
                     i,
                     clusterPos[i].w,
                     minAllowedNormalization);
            clusterPos[i].x = 0.0;
            clusterPos[i].y = 0.0;
            clusterPos[i].z = 0.0;
            //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
          }
        }

        float diff2 = 0.0;
        for (int s = 0; s < nSeeds; s++) {
          int i = getSeedRhIdx(s);
          float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
          if (debug)
            printf("\tCluster %d (seed %d) has delta2 = %f\n", s, getSeedRhIdx(s), delta2);
          if (delta2 > diff2) {
            diff2 = delta2;
            if (debug)
              printf("\t\tNew diff2 = %f\n", diff2);
          }
        }

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
      if (iter > 1) {
        if (iter >= maxIterations)
          printf("topoId %d (nSeeds = %d  nRHTopo = %d) hit maxIterations (%d) with diff (%f) > tol (%f)\n",
                 topoId,
                 nSeeds,
                 nRHTopo,
                 iter,
                 diff,
                 stoppingTolerance * tolScaling);
        else
          printf("topoId %d converged in %d iterations\n", topoId, iter);
      }
    }
  }

  __global__ void hcalFastCluster_step1(size_t size,
                                        const float* __restrict__ pfrh_x,
                                        const float* __restrict__ pfrh_y,
                                        const float* __restrict__ pfrh_z,
                                        const float* __restrict__ pfrh_energy,
                                        int* pfrh_topoId,
                                        int* pfrh_isSeed,
                                        const int* __restrict__ pfrh_layer,
                                        const int* __restrict__ pfrh_depth,
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

        if (pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
          fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
        } else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) {
          fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
        }

        if (fraction == -1.)
          printf("FRACTION is NEGATIVE!!!");

        if (pfrh_isSeed[j] != 1) {
          atomicAdd(&fracSum[j], fraction);
        }
      }
    }
  }

  __global__ void hcalFastCluster_step2(size_t size,
                                        const float* __restrict__ pfrh_x,
                                        const float* __restrict__ pfrh_y,
                                        const float* __restrict__ pfrh_z,
                                        const float* __restrict__ pfrh_energy,
                                        int* pfrh_topoId,
                                        int* pfrh_isSeed,
                                        const int* __restrict__ pfrh_layer,
                                        const int* __restrict__ pfrh_depth,
                                        float* pcrhfrac,
                                        int* pcrhfracind,
                                        float* fracSum,
                                        int* rhCount,
                                        int* topoSeedCount,
                                        int* topoRHCount,
                                        int* seedFracOffsets,
                                        int* topoSeedOffsets,
                                        int* topoSeedList) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if (i < size && j < size) {
      if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
        if (i == j) {
          pcrhfrac[i * 100] = 1.;
          pcrhfracind[i * 100] = j;
        }
        if (pfrh_isSeed[j] != 1) {
          float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                        (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                        (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

          float d2 = dist2 / showerSigma2;
          float fraction = -1.;

          if (pfrh_layer[j] == 1) {
            fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
          } else if (pfrh_layer[j] == 3) {
            fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
          }

          if (fraction == -1.)
            printf("FRACTION is NEGATIVE!!!");

          if (fracSum[j] > minFracTot) {
            float fracpct = fraction / fracSum[j];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
              int k = atomicAdd(&rhCount[i], 1);
              pcrhfrac[seedFracOffsets[i] + k] = fracpct;
              pcrhfracind[seedFracOffsets[i] + k] = j;
              //pcrhfrac[i*100+k] = fracpct;
              //pcrhfracind[i*100+k] = j;
            }
          }
          /*
        if(d2 < 100. )
          {
            if ((fraction/fracSum[j])>minFracToKeep){
              int k = atomicAdd(&rhCount[i],1);
              pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
              pcrhfracind[i*maxSize+k] = j;
              //printf("(i,j)=(%d,%d), rhCount=%d, fraction=%f, fracsum=%f \n",i,j,rhCount[i], fraction, fracSum[j]);
            }
          }
        */
        }
      }
    }
  }

  __global__ void hcalFastCluster_step2(size_t size,
                                        const float* __restrict__ pfrh_x,
                                        const float* __restrict__ pfrh_y,
                                        const float* __restrict__ pfrh_z,
                                        const float* __restrict__ pfrh_energy,
                                        int* pfrh_topoId,
                                        int* pfrh_isSeed,
                                        const int* __restrict__ pfrh_layer,
                                        const int* __restrict__ pfrh_depth,
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
          pcrhfrac[i * 100] = 1.;
          pcrhfracind[i * 100] = j;
        }
        if (pfrh_isSeed[j] != 1) {
          float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                        (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                        (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

          float d2 = dist2 / showerSigma2;
          float fraction = -1.;

          if (pfrh_layer[j] == 1) {
            fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
          } else if (pfrh_layer[j] == 3) {
            fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
          }

          if (fraction == -1.)
            printf("FRACTION is NEGATIVE!!!");

          if (fracSum[j] > minFracTot) {
            float fracpct = fraction / fracSum[j];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
              int k = atomicAdd(&rhCount[i], 1);
              pcrhfrac[i * 100 + k] = fracpct;
              pcrhfracind[i * 100 + k] = j;
            }
          }
          /*
        if(d2 < 100. )
          {
            if ((fraction/fracSum[j])>minFracToKeep){
              int k = atomicAdd(&rhCount[i],1);
              pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
              pcrhfracind[i*maxSize+k] = j;
              //printf("(i,j)=(%d,%d), rhCount=%d, fraction=%f, fracsum=%f \n",i,j,rhCount[i], fraction, fracSum[j]);
            }
          }
        */
        }
      }
    }
  }

  __global__ void hcalFastCluster_step1_serialize(size_t size,
                                                  const float* __restrict__ pfrh_x,
                                                  const float* __restrict__ pfrh_y,
                                                  const float* __restrict__ pfrh_z,
                                                  const float* __restrict__ pfrh_energy,
                                                  int* pfrh_topoId,
                                                  int* pfrh_isSeed,
                                                  const int* __restrict__ pfrh_layer,
                                                  const int* __restrict__ pfrh_depth,
                                                  float* pcrhfrac,
                                                  int* pcrhfracind,
                                                  float* fracSum,
                                                  int* rhCount) {
    //int i = threadIdx.x+blockIdx.x*blockDim.x;
    //int j = threadIdx.y+blockIdx.y*blockDim.y;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        //make sure topoID, Layer is the same, i is seed and j is not seed
        if (i < size && j < size) {
          if (pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i] == 1) {
            float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                          (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                          (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

            float d2 = dist2 / showerSigma2;
            float fraction = -1.;

            if (pfrh_layer[j] == 1) {
              fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
            } else if (pfrh_layer[j] == 3) {
              fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
            }

            if (fraction == -1.)
              printf("FRACTION is NEGATIVE!!!");

            if (pfrh_isSeed[j] != 1) {
              atomicAdd(&fracSum[j], fraction);
            }
          }
        }
      }
    }
  }

  __global__ void hcalFastCluster_step2_serialize(size_t size,
                                                  const float* __restrict__ pfrh_x,
                                                  const float* __restrict__ pfrh_y,
                                                  const float* __restrict__ pfrh_z,
                                                  const float* __restrict__ pfrh_energy,
                                                  int* pfrh_topoId,
                                                  int* pfrh_isSeed,
                                                  const int* __restrict__ pfrh_layer,
                                                  const int* __restrict__ pfrh_depth,
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
              pcrhfrac[i * 100] = 1.;
              pcrhfracind[i * 100] = j;
            }
            if (pfrh_isSeed[j] != 1) {
              float dist2 = (pfrh_x[i] - pfrh_x[j]) * (pfrh_x[i] - pfrh_x[j]) +
                            (pfrh_y[i] - pfrh_y[j]) * (pfrh_y[i] - pfrh_y[j]) +
                            (pfrh_z[i] - pfrh_z[j]) * (pfrh_z[i] - pfrh_z[j]);

              float d2 = dist2 / showerSigma2;
              float fraction = -1.;

              if (pfrh_layer[j] == 1) {
                fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
              } else if (pfrh_layer[j] == 3) {
                fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2);
              }

              if (fraction == -1.)
                printf("FRACTION is NEGATIVE!!!");
              if (d2 < 100.) {
                if ((fraction / fracSum[j]) > minFracToKeep) {
                  int k = atomicAdd(&rhCount[i], 1);
                  pcrhfrac[i * 100 + k] = fraction / fracSum[j];
                  pcrhfracind[i * 100 + k] = j;
                  //printf("(i,j)=(%d,%d), rhCount=%d, fraction=%f, fracsum=%f \n",i,j,rhCount[i], fraction, fracSum[j]);
                }
              }
            }
          }
        }
      }
    }
  }

  // Compute whether rechits pass topo clustering energy threshold
  __global__ void passingTopoThreshold(size_t size,
                                       const int* __restrict__ pfrh_layer,
                                       const int* __restrict__ pfrh_depth,
                                       const float* __restrict__ pfrh_energy,
                                       bool* pfrh_passTopoThresh) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
      if ((pfrh_layer[i] == 3 && pfrh_energy[i] > topoEThresholdEE_vec[pfrh_depth[i] - 1]) ||
          (pfrh_layer[i] == 1 && pfrh_energy[i] > topoEThresholdEB_vec[pfrh_depth[i] - 1])) {
        pfrh_passTopoThresh[i] = true;
      } else {
        pfrh_passTopoThresh[i] = false;
      }
    }
  }

  __global__ void passingTopoThreshold(int size,
                                       const int* __restrict__ pfrh_layer,
                                       const int* __restrict__ pfrh_depth,
                                       const float* __restrict__ pfrh_energy,
                                       bool* pfrh_passTopoThresh) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
      if ((pfrh_layer[i] == 3 && pfrh_energy[i] > topoEThresholdEE_vec[pfrh_depth[i] - 1]) ||
          (pfrh_layer[i] == 1 && pfrh_energy[i] > topoEThresholdEB_vec[pfrh_depth[i] - 1])) {
        pfrh_passTopoThresh[i] = true;
      } else {
        pfrh_passTopoThresh[i] = false;
      }
    }
  }

  // Contraction in a single block
  __global__ void topoClusterContraction(size_t size,
                                         int* pfrh_parent,
                                         int* pfrh_isSeed,
                                         //const int* pfrh_neighbours, // temporary inputs for debuggi
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

    // // debugging printing block
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //   int nnode=0;
    //   for (int i = 0; i < size; i++) {
    // 	//printf("final pfrh_id,parent: %d %d\n",i,pfrh_parent[i]);
    // 	if (i==pfrh_parent[i]) nnode++;
    //   }
    //   printf("pfrh_parent 3 multiplicity: %d\n",nnode);
    //   for (int pos = 0; pos < size; pos++) {
    // 	int parent_target = pfrh_parent[pos];
    // 	for (int i = 0; i < 8; i++) {
    // 	  int neighbor_id = pfrh_neighbours[pos * 8 + i];
    // 	  if (neighbor_id>-1){ // valid neighbors
    // 	    int parent_neighbor = pfrh_parent[neighbor_id];
    // 	    if (parent_target!=parent_neighbor){
    // 	      printf("hmm. they should have the same parent, but they don't. why... %d %d\n",pos,neighbor_id);
    // 	    }
    // 	  }
    // 	}
    //   }
    //   }
    // __syncthreads();

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
      if (*pcrhFracSize>200000) // DeclsForKernels.h maxPFCFracs
	printf("At the end of topoClusterContraction, found large *pcrhFracSize = %d\n", *pcrhFracSize);
    }

  }

  // Prefill the rechit index for all PFCluster fractions
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

  __global__ void fillRhfIndex_serialize(size_t nRH,
                                         int* pfrh_parent,
                                         int* pfrh_isSeed,
                                         int* topoSeedCount,
                                         int* topoRHCount,
                                         int* seedFracOffsets,
                                         int* rhCount,
                                         int* pcrhfracind) {
    //int debugSeedIdx = 500;

    /*
    printf("rhCount = \n[");
    for (int i = 0; i < (int)nRH; i++) {
        if (i != 0) printf(", ");
        printf("%d", rhCount[i]);
    }
    printf("]\n");
    */

    for (int i = 0; i < (int)nRH; i++) {
      for (int j = 0; j < (int)nRH; j++) {
        //if (i == debugSeedIdx) printf("This is fillRhfIndex with i = %d and j = %d\n", i, j);
        int topoId = pfrh_parent[i];
        if (topoId == pfrh_parent[j] && topoId > -1 && pfrh_isSeed[i] && !pfrh_isSeed[j]) {
          //if (i == debugSeedIdx) printf("This is seed %d with topoId %d and rechit %d\n", i, topoId, j);
          int k = atomicAdd(&rhCount[i], 1);  // Increment the number of rechit fractions for this seed
          if (seedFracOffsets[i] < 0)
            printf("WARNING: seed %d has offset %d!\n", i, seedFracOffsets[i]);
          //printf("seed %d: rechit %d index with k = %d and seed offset = %d\n", i, j, k, seedFracOffsets[i]);
          pcrhfracind[seedFracOffsets[i] + k] = j;  // Save this rechit index
        }
      }
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

  // when on the left edge of the edgeId/List block, returns true
  __device__ __forceinline__ bool isLeftEdgeKH(const int idx,
                                               const int nEdges,
                                               const int* __restrict__ pfrh_edgeId,
                                               const int* __restrict__ pfrh_edgeMask) {
    int temp = idx - 1;
    if (idx > 0) {
      int edgeId = pfrh_edgeId[idx];
      int tempId = pfrh_edgeId[temp];
      if (edgeId != tempId) {
	// Different topo Id here!
	return true;
      }
    } else if (temp < 0) { // idx==0
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
                                     int* nEdgesIn,
                                     int* pfrh_parent,
                                     int* pfrh_edgeId,
                                     int* pfrh_edgeList,
                                     int* pfrh_edgeMask,
                                     const int* pfrh_passTopoThresh,
                                     int* topoIter) {
    __shared__ int notDone; // This is better be bool, but somehow it leads to out of bound
    __shared__ int iter, gridStride, nEdges;

    int start = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
      *topoIter = 0;
      iter = 0;
      nEdges = *nEdgesIn;
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

    do {
      if (threadIdx.x == 0) {
        notDone = 0;
      }
      __syncthreads();

      // Odd linking
      for (int idx = start; idx < nEdges; idx += gridStride) {
        int i = pfrh_edgeId[idx];  // Get edge topo id
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
            notDone = 1;
          } else {
            pfrh_edgeMask[idx] = 0;
          }
        }
      }
      if (threadIdx.x == 0)
        iter++;

      __syncthreads();

      if (notDone==0)
        break;

      __syncthreads();//!!

      if (threadIdx.x == 0) {
        notDone = 0;
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
            notDone = 1;
          } else {
            pfrh_edgeMask[idx] = 0;
          }
        }
      }

      if (threadIdx.x == 0)
        iter++;

      __syncthreads();

    } while (notDone==1);

    *topoIter = iter;
#ifdef DEBUG_GPU_HCAL
//    if (threadIdx.x == 0) {
//        printf("*** Topo clustering converged in %d iterations ***\n", iter);
//    }
//    __syncthreads();
#endif
  }

  __global__ void topoClusterLinkingKH(int nRH,
                                       int* nEdgesIn,
                                       //float* pfrh_energy, // Temporary entry for debugging
                                       int* pfrh_parent,
                                       int* pfrh_edgeId,
                                       int* pfrh_edgeList,
                                       int* pfrh_edgeMask,
                                       const int* pfrh_passTopoThresh,
                                       int* topoIter) {
    __shared__ int notDone;  // This is better be bool, but somehow it leads to out of bound
    __shared__ int notDone2;
    __shared__ int gridStride, nEdges;

    // Initialization
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
      *topoIter = 0;
      nEdges = *nEdgesIn;
      gridStride = blockDim.x * gridDim.x;  // For single block kernel this is the number of threads
    }

    __syncthreads();

    // Check if pairs in edgeId,edgeList contain a rh not passing topo threshold
    // If found, set the mask to 0
    // But, for now, not using edgeMask hereafter, because the same threshold cut is applied at the PFRecHit level
    // for (int idx = start; idx < nEdges; idx += gridStride) {
    //   if (pfrh_passTopoThresh[pfrh_edgeId[idx]] && pfrh_passTopoThresh[pfrh_edgeList[idx]])
    //     pfrh_edgeMask[idx] = 1;
    //   else
    //     pfrh_edgeMask[idx] = 0;
    // }

    // __syncthreads();

    // // Print out debugging info
    // if (threadIdx.x == 0) {
    //   // for (int idx = 0; idx < nEdges; idx++) {
    //   // 	printf("initial edge id, list, mask: %d %d %d\n",pfrh_edgeId[idx],pfrh_edgeList[idx],pfrh_edgeMask[idx]);
    //   // 	//printf("initial edge id, list, mask: %d %d\n",pfrh_edgeId[idx],pfrh_edgeList[idx]);
    //   //  }
    //   printf("number of eges %d\n",nEdges);
    //   // for (int i = 0; i < nRH; i++) {
    //   // 	printf("initial pfrh_id,parent,energy: %d %d %8.3f\n",i,pfrh_parent[i],pfrh_energy[i]);
    //   // }
    // }

    // Explicitly initialize pfrh_parent
    for (int i = start; i < nRH; i += gridStride) {
      pfrh_parent[i] = i;
    }

    __syncthreads();

    // for notDone
    if (threadIdx.x == 0) {
      notDone = 0;
      notDone2 = 0;
      //printf("gridStride, blockDim.x %d %d\n",gridStride,blockDim.x);
    }

    __syncthreads();

    // (1) First attempt
    // First edge [set parents to those smaller numbers]
    for (int idx = start; idx < nEdges; idx += gridStride) {
      int i = pfrh_edgeId[idx];  // Get edge topo id
      if (pfrh_edgeMask[idx] > 0 && isLeftEdgeKH(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) { // isLeftEdgeKH
	pfrh_parent[i] = (int)min(i, pfrh_edgeList[idx]);
      }
    }

    __syncthreads();

    // KenH
    for (int ii=0; ii<100; ii++) { // loop until topo clustering iteration converges

      // for notDone
      if (threadIdx.x == 0) {
    	notDone2 = 0;
      }

      // Follow parents of parents .... to contract parent structure
      do {
    	volatile bool threadNotDone = false;
    	for (int i = threadIdx.x; i < nRH; i += blockDim.x) {
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

      __syncthreads();

      // Print out debugging info
      // Connect remaining links
      // if (threadIdx.x == 0) {
      // 	int nnode=0;
      // 	for (int i = 0; i < nRH; i++) {
      // 	  //printf("middle pfrh_id,parent: %d %d\n",i,pfrh_parent[i]);
      // 	  if (i==pfrh_parent[i]) nnode++;
      // 	}
      // 	printf("pfrh_parent multiplicity: %d\n",nnode);
      // }

      // __syncthreads();

      for (int idx = start; idx < nEdges; idx += gridStride) {
      	//for (int idx = 0; idx < nEdges; idx++) {
      	int i = pfrh_edgeId[idx];    // Get edge topo id
      	int j = pfrh_edgeList[idx];  // Get edge neighbor list
      	int parent_target = pfrh_parent[i];
      	int parent_neighbor = pfrh_parent[j];
      	if (parent_target!=parent_neighbor){
      	  notDone2 = 1;
      	  //printf("hmm. they should have the same parent, but they don't. why... %d %d %d\n",i,j,ii);
      	  int min_parent = (int)min(parent_target,parent_neighbor);
      	  int max_parent = (int)max(parent_target,parent_neighbor);
      	  int idx_max = i;
      	  if (parent_neighbor == max_parent) idx_max = j;
      	  pfrh_parent[idx_max] = min_parent;
      	}
      }

      __syncthreads();
      if (notDone2==0) // if topocluster finding is converged, terminate the for-ii loop
    	break;

    } // for-loop ii

    __syncthreads();

    // Follow parents of parents .... to contract parent structure
    do {
      volatile bool threadNotDone = false;
      for (int i = threadIdx.x; i < nRH; i += blockDim.x) {
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

    // __syncthreads();

    // // Print out debugging info
    // if (threadIdx.x == 0) {
    //   int nnode=0;
    //   for (int i = 0; i < nRH; i++) {
    // 	//printf("middle2 pfrh_id,parent: %d %d\n",i,pfrh_parent[i]);
    // 	if (i==pfrh_parent[i]) nnode++;
    //   }
    //   printf("pfrh_parent 2 multiplicity: %d\n",nnode);
    // }

  }

  __device__ __forceinline__ void sortSwap(int* toSort, int a, int b) {
    const int tmp = min(toSort[a], toSort[b]);
    toSort[b] = max(toSort[a], toSort[b]);
    toSort[a] = tmp;
  }

  __device__ __forceinline__ void sortEight(int* toSort, int* nEdges) {
    // toSort is a pointer to shared mem neighbor list. 8 elements should be accessed
    // nEdges is the number of valid neighbors for this rechit
    // Sorting network optimized for 8 elements
    sortSwap(toSort, 0, 4);
    sortSwap(toSort, 1, 5);
    sortSwap(toSort, 2, 6);
    sortSwap(toSort, 3, 7);

    sortSwap(toSort, 0, 2);
    sortSwap(toSort, 1, 3);
    sortSwap(toSort, 4, 6);
    sortSwap(toSort, 5, 7);

    sortSwap(toSort, 2, 4);
    sortSwap(toSort, 3, 5);

    sortSwap(toSort, 0, 1);
    sortSwap(toSort, 2, 3);
    sortSwap(toSort, 4, 5);
    sortSwap(toSort, 6, 7);

    sortSwap(toSort, 1, 4);
    sortSwap(toSort, 3, 6);

    sortSwap(toSort, 1, 2);
    sortSwap(toSort, 3, 4);
    sortSwap(toSort, 5, 6);

    // Add total number of edges
    // min/max eliminates warp divergences in ternary operation
    *nEdges += max(min(toSort[0] + 1, 1), 0);
    *nEdges += max(min(toSort[1] + 1, 1), 0);
    *nEdges += max(min(toSort[2] + 1, 1), 0);
    *nEdges += max(min(toSort[3] + 1, 1), 0);
    *nEdges += max(min(toSort[4] + 1, 1), 0);
    *nEdges += max(min(toSort[5] + 1, 1), 0);
    *nEdges += max(min(toSort[6] + 1, 1), 0);
    *nEdges += max(min(toSort[7] + 1, 1), 0);

    //    *nEdges += (toSort[0] > -1 ? 1 : 0);
    //    *nEdges += (toSort[1] > -1 ? 1 : 0);
    //    *nEdges += (toSort[2] > -1 ? 1 : 0);
    //    *nEdges += (toSort[3] > -1 ? 1 : 0);
    //    *nEdges += (toSort[4] > -1 ? 1 : 0);
    //    *nEdges += (toSort[5] > -1 ? 1 : 0);
    //    *nEdges += (toSort[6] > -1 ? 1 : 0);
    //    *nEdges += (toSort[7] > -1 ? 1 : 0);
  }

  __device__ __forceinline__ int scan1Inclusive(int idata, volatile int* s_Data, int size) {
    int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (int offset = 1; offset < size; offset <<= 1) {
      int t = s_Data[pos] + s_Data[pos - offset];
      s_Data[pos] = t;
    }

    return s_Data[pos];
  }

  __device__ __forceinline__ int scan1Exclusive(int idata, volatile int* s_Data, int size) {
    return scan1Inclusive(idata, s_Data, size) - idata;
  }

  __device__ __forceinline__ int4 scan4Inclusive(int4 idata4, volatile int* s_Data, int size) {
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    int oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
  }

  //Exclusive vector scan: the array to be scanned is stored
  //in local thread memory scope as uint4
  __device__ __forceinline__ int4 scan4Exclusive(int4 idata4, volatile int* s_Data, int size) {
    int4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
  }

  __global__ void prepareTopoInputsSerial(int nRH,
                                          int* nEdges,
                                          const int* pfrh_passTopoThresh,
                                          const int* pfrh_neighbours,
                                          int* pfrh_edgeId,
                                          int* pfrh_edgeList) {
    extern __shared__ int smem[];
    int* sortingArray = smem;
    int* nEdgeArray = (int*)&sortingArray[blockDim.x * 8];  // Number of edges after sorting
    //int* nEdgeArraySummed = (int*)&nEdgeArray[blockDim.x];  // Cumulative sum of number of edges after sorting
    //int* s_Data = (int*)&nEdgeArraySummed[blockDim.x];  // temp buffer of length 2*THREADBLOCK_SIZE
    __shared__ int total;

    if (threadIdx.x == 0) {
      total = 0;
      *nEdges = 0;
    }
    __syncthreads();

    for (int pos = 0; pos < nRH; pos++) {
      nEdgeArray[0] = 0;
      for (int i = 0; i < 8; i++) {
        sortingArray[i] = pfrh_neighbours[pos * 8 + i];
      }
      sortEight(sortingArray, nEdgeArray);

      for (unsigned p = 0; p < nEdgeArray[0]; p++) {
        pfrh_edgeId[total + p] = pos;
        pfrh_edgeList[total + p] = sortingArray[8 - nEdgeArray[0] + p];
      }
      total += nEdgeArray[0];
    }

    *nEdges = total;
    return;
  }

  // Kernel to fill pfrh_edgeId, pfrh_edgeList arrays used in topo clustering
  __global__ void prepareTopoInputs(int nRH,
                                    int* nEdges,
                                    const int* pfrh_passTopoThresh,
                                    const int* pfrh_neighbours,
                                    int* pfrh_edgeId,
                                    int* pfrh_edgeList) {
    extern __shared__ int smem[];
    int* sortingArray = smem;                               // Buffer to store neighbor lists
    int* nEdgeArray = (int*)&sortingArray[blockDim.x * 8];  // Number of edges after sorting
    int* nEdgeArraySummed = (int*)&nEdgeArray[blockDim.x];  // Cumulative sum of number of edges after sorting
    int* s_Data = (int*)&nEdgeArraySummed[blockDim.x];      // Temp buffer of length 2*THREADBLOCK_SIZE
    __shared__ int total, maxIter;

    if (threadIdx.x == 0) {
      total = 0;
      maxIter = (nRH + blockDim.x - 1) / blockDim.x;  // Fast ceiling of nRH/blockDim.x
    }
    __syncthreads();

    int pos = 0;
    for (int iter = 0; iter < maxIter - 1; iter++) {
      nEdgeArray[threadIdx.x] = 0;
      nEdgeArraySummed[threadIdx.x] = 0;
      //if (threadIdx.x == 0) printf("\n--- Now on iter % d ---", iter);
      __syncthreads();

      pos = iter * blockDim.x * 8;
      // Load values into shared mem using coalesced reads
      sortingArray[threadIdx.x] = pfrh_neighbours[pos + threadIdx.x];
      sortingArray[blockDim.x + threadIdx.x] = pfrh_neighbours[pos + blockDim.x + threadIdx.x];
      sortingArray[2 * blockDim.x + threadIdx.x] = pfrh_neighbours[pos + 2 * blockDim.x + threadIdx.x];
      sortingArray[3 * blockDim.x + threadIdx.x] = pfrh_neighbours[pos + 3 * blockDim.x + threadIdx.x];
      sortingArray[4 * blockDim.x + threadIdx.x] = pfrh_neighbours[pos + 4 * blockDim.x + threadIdx.x];
      sortingArray[5 * blockDim.x + threadIdx.x] = pfrh_neighbours[pos + 5 * blockDim.x + threadIdx.x];
      sortingArray[6 * blockDim.x + threadIdx.x] = pfrh_neighbours[pos + 6 * blockDim.x + threadIdx.x];
      sortingArray[7 * blockDim.x + threadIdx.x] = pfrh_neighbours[pos + 7 * blockDim.x + threadIdx.x];

      __syncthreads();
      //        if (threadIdx.x == 0) {
      //            printf("\nsortingArray:\n");
      //            for (int i = 0; i < blockDim.x; i++) {
      //                for (int j = 0; j < 8; j++)
      //                    printf("%d ", sortingArray[i * 8 + j]);
      //                printf("\n");
      //            }
      //            printf("\n");
      //        }

      // Sort rechit neighbors
      sortEight(sortingArray + 8 * threadIdx.x, nEdgeArray + threadIdx.x);

      //        if (threadIdx.x == 0) {
      //            printf("\nnEdgeArray:\n");
      //            for (int i = 0; i < blockDim.x; i++)
      //                printf("%d ", nEdgeArray[i]);
      //            printf("\n");
      //
      //            printf("\nnEdgeArraySummed:\n");
      //            for (int i = 0; i < blockDim.x; i++)
      //                printf("%d ", nEdgeArraySummed[i]);
      //            printf("\n");
      //
      //            printf("\ns_Data:\n");
      //            for (int i = 0; i < 2 * blockDim.x; i++)
      //                printf("%d ", s_Data[i]);
      //            printf("\n\n");
      //        }
      __syncthreads();

      // Exclusive parallel scan to cumulatively sum number of neighbors for each rechit
      // Used as offsets for filling edgeId, edgeList arrays
      if (threadIdx.x < blockDim.x / 4) {
        int edgePos = threadIdx.x * 4;
        //Load data
        int4 idata4 =
            make_int4(nEdgeArray[edgePos], nEdgeArray[edgePos + 1], nEdgeArray[edgePos + 2], nEdgeArray[edgePos + 3]);

        //Calculate exclusive scan
        int4 odata4 = scan4Exclusive(idata4, s_Data, blockDim.x);

        //Write back
        nEdgeArraySummed[edgePos] = odata4.x;
        nEdgeArraySummed[edgePos + 1] = odata4.y;
        nEdgeArraySummed[edgePos + 2] = odata4.z;
        nEdgeArraySummed[edgePos + 3] = odata4.w;
      }
      __syncthreads();

      //        if (threadIdx.x == 0) {
      //            printf("\nsortingArray:\n");
      //            for (int i = 0; i < blockDim.x; i++) {
      //                for (int j = 0; j < 8; j++)
      //                    printf("%d ", sortingArray[i * 8 + j]);
      //                printf("\n");
      //            }
      //
      //            printf("\nnEdgeArray:\n");
      //            for (int i = 0; i < blockDim.x; i++)
      //                printf("%d ", nEdgeArray[i]);
      //            printf("\n");
      //
      //            printf("\nnEdgeArraySummed:\n");
      //            for (int i = 0; i < blockDim.x; i++)
      //                printf("%d ", nEdgeArraySummed[i]);
      //            printf("\n");
      //
      //            printf("\ns_Data:\n");
      //            for (int i = 0; i < 2 * blockDim.x; i++)
      //                printf("%d ", s_Data[i]);
      //            printf("\n\n");
      //        }
      //        __syncthreads();

      // Fill edgeId, edgeList arrays
      for (unsigned p = 0; p < nEdgeArray[threadIdx.x]; p++) {
        pfrh_edgeId[total + nEdgeArraySummed[threadIdx.x] + p] = blockDim.x * iter + threadIdx.x;
        pfrh_edgeList[total + nEdgeArraySummed[threadIdx.x] + p] =
            sortingArray[8 * threadIdx.x + 8 - nEdgeArray[threadIdx.x] + p];
      }
      __syncthreads();

      // Add sum of edges to the total
      if (threadIdx.x == 0) {
        atomicAdd(&total, nEdgeArraySummed[blockDim.x - 1] + nEdgeArray[blockDim.x - 1]);
        //            printf("\nTotal: %d\n", total);
        //
        //            printf("\nedgeId:\n");
        //            for (int i = 0; i < total; i++)
        //                printf("%d ", edgeId[i]);
        //            printf("\n");
        //
        //            printf("\nedgeList:\n");
        //            for (int i = 0; i < total; i++)
        //                printf("%d ", edgeList[i]);
        //            printf("\n");

        //printf("Total sum: %d\n", nEdgeArraySummed[blockDim.x - 1] + nEdgeArray[blockDim.x - 1]);
      }
      __syncthreads();
    }
    // Final iteration. Remaining rechits <= blockDim.x
    int rhIdx = blockDim.x * (maxIter - 1) + threadIdx.x;

    nEdgeArray[threadIdx.x] = 0;
    nEdgeArraySummed[threadIdx.x] = 0;
    //if (threadIdx.x == 0) printf("\n--- Now on iter %d (final) ---\n", maxIter-1);
    __syncthreads();

    pos = (maxIter - 1) * blockDim.x * 8;
    if (rhIdx < nRH) {
      //printf("Thread %d on rhIdx = %d\n", threadIdx.x, rhIdx);
      // Load values into shared mem
      sortingArray[8 * threadIdx.x] = pfrh_neighbours[pos + 8 * threadIdx.x];
      sortingArray[8 * threadIdx.x + 1] = pfrh_neighbours[pos + 8 * threadIdx.x + 1];
      sortingArray[8 * threadIdx.x + 2] = pfrh_neighbours[pos + 8 * threadIdx.x + 2];
      sortingArray[8 * threadIdx.x + 3] = pfrh_neighbours[pos + 8 * threadIdx.x + 3];
      sortingArray[8 * threadIdx.x + 4] = pfrh_neighbours[pos + 8 * threadIdx.x + 4];
      sortingArray[8 * threadIdx.x + 5] = pfrh_neighbours[pos + 8 * threadIdx.x + 5];
      sortingArray[8 * threadIdx.x + 6] = pfrh_neighbours[pos + 8 * threadIdx.x + 6];
      sortingArray[8 * threadIdx.x + 7] = pfrh_neighbours[pos + 8 * threadIdx.x + 7];
    } else {
      //printf("Skipping thread %d (rhIdx = %d: out of bounds)\n", threadIdx.x, rhIdx);
      // These threads exceed length of rechits
      sortingArray[8 * threadIdx.x] = -1;
      sortingArray[8 * threadIdx.x + 1] = -1;
      sortingArray[8 * threadIdx.x + 2] = -1;
      sortingArray[8 * threadIdx.x + 3] = -1;
      sortingArray[8 * threadIdx.x + 4] = -1;
      sortingArray[8 * threadIdx.x + 5] = -1;
      sortingArray[8 * threadIdx.x + 6] = -1;
      sortingArray[8 * threadIdx.x + 7] = -1;
    }
    __syncthreads();

    //    if (threadIdx.x == 0) {
    //        printf("\nsortingArray:\n");
    //        for (int i = 0; i < blockDim.x; i++) {
    //            for (int j = 0; j < 8; j++)
    //                printf("%d ", sortingArray[i * 8 + j]);
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }

    sortEight(sortingArray + 8 * threadIdx.x, nEdgeArray + threadIdx.x);

    //    if (threadIdx.x == 0) {
    //        printf("\nAfter sortEight, sortingArray:\n\n");
    //        for (int i = 0; i < (nRH - blockDim.x * (maxIter-1)); i++) {
    //            printf("rh %d:\t", i + blockDim.x * (maxIter-1));
    //            for (int j = 0; j < 8; j++) {
    //                if (j > 0) printf(", ");
    //                printf("%d", sortingArray[8*i + j]);
    //            }
    //            printf("\n");
    //        }
    //    }
    __syncthreads();

    //    if (threadIdx.x == 0) {
    //        printf("\nnEdgeArray:\n");
    //        for (int i = 0; i < blockDim.x; i++)
    //            printf("%d ", nEdgeArray[i]);
    //        printf("\n");
    //
    //        printf("\nnEdgeArraySummed:\n");
    //        for (int i = 0; i < blockDim.x; i++)
    //            printf("%d ", nEdgeArraySummed[i]);
    //        printf("\n");
    //
    //        printf("\ns_Data:\n");
    //        for (int i = 0; i < 2 * blockDim.x; i++)
    //            printf("%d ", s_Data[i]);
    //        printf("\n\n");
    //    }
    //    __syncthreads();

    if (threadIdx.x < blockDim.x / 4) {
      int edgePos = threadIdx.x * 4;
      //Load data
      int4 idata4 =
          make_int4(nEdgeArray[edgePos], nEdgeArray[edgePos + 1], nEdgeArray[edgePos + 2], nEdgeArray[edgePos + 3]);

      //Calculate exclusive scan
      int4 odata4 = scan4Exclusive(idata4, s_Data, blockDim.x);

      //Write back
      nEdgeArraySummed[edgePos] = odata4.x;
      nEdgeArraySummed[edgePos + 1] = odata4.y;
      nEdgeArraySummed[edgePos + 2] = odata4.z;
      nEdgeArraySummed[edgePos + 3] = odata4.w;
    }
    __syncthreads();

    //    if (threadIdx.x == 0) {
    //        printf("\nsortingArray:\n");
    //        for (int i = 0; i < blockDim.x; i++) {
    //            for (int j = 0; j < 8; j++)
    //                printf("%d ", sortingArray[i * 8 + j]);
    //            printf("\n");
    //        }
    //
    //        printf("\nnEdgeArray:\n[");
    //        for (int i = 0; i < blockDim.x; i++) {
    //          if (i > 0) printf(", ");
    //          printf("%d", nEdgeArray[i]);
    //        }
    //        printf("]\n\n");
    //
    //        printf("\nnEdgeArraySummed:\n");
    //        for (int i = 0; i < blockDim.x; i++)
    //            printf("%d ", nEdgeArraySummed[i]);
    //        printf("\n");
    //
    //        printf("\ns_Data:\n");
    //        for (int i = 0; i < 2 * blockDim.x; i++)
    //            printf("%d ", s_Data[i]);
    //        printf("\n\n");
    //    }
    //    __syncthreads();

    // Fill edgeId, edgeList arrays
    for (unsigned p = 0; p < nEdgeArray[threadIdx.x]; p++) {
      pfrh_edgeId[total + nEdgeArraySummed[threadIdx.x] + p] = blockDim.x * (maxIter - 1) + threadIdx.x;
      pfrh_edgeList[total + nEdgeArraySummed[threadIdx.x] + p] =
          sortingArray[8 * threadIdx.x + 8 - nEdgeArray[threadIdx.x] + p];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicAdd(&total, nEdgeArraySummed[blockDim.x - 1] + nEdgeArray[blockDim.x - 1]);

      //        printf("\nnEdgeArray:\n[");
      //        for (int i = 0; i < blockDim.x; i++) {
      //          if (i > 0) printf(", ");
      //          printf("%d", nEdgeArray[i]);
      //        }
      //        printf("]\n\n");
      //
      //        printf("\nnEdgeArraySummed:\n[");
      //        for (int i = 0; i < blockDim.x; i++) {
      //          if (i > 0) printf(", ");
      //          printf("%d", nEdgeArraySummed[i]);
      //        }
      //        printf("]\n\n");

      //        printf("\nTotal: %d\n", total);
      //
      //        printf("\nedgeId:\n");
      //        for (int i = 0; i < total; i++)
      //            printf("%d ", edgeId[i]);
      //        printf("\n");
      //
      //        printf("\nedgeList:\n");
      //        for (int i = 0; i < total; i++)
      //            printf("%d ", edgeList[i]);
      //        printf("\n");

      //printf("Total sum: %d\n", nEdgeArraySummed[blockDim.x - 1] + nEdgeArray[blockDim.x - 1]);
    }
    __syncthreads();

    if (threadIdx.x == 0)
      *nEdges = total;
  }

  // Debugging kernel to compare CPU & GPU edge arrays
  __global__ void compareEdgeArrays(int* gpu_nEdges,
                                    int* gpu_edgeId,
                                    int* gpu_edgeList,
                                    int cpu_nEdges,
                                    int* cpu_edgeId,
                                    int* cpu_edgeList,
                                    int nRH,
                                    int* cpu_neigh4List,
                                    int* pfrh_neighbours) {
    printf("Now in compareEdgeArrays\n");

    //    printf("\n--- gpu_edgeId ---\n[");
    //    for (int i = 0; i < *gpu_nEdges; i++) {
    //        if (i > 0) printf(", ");
    //        printf("%d", gpu_edgeId[i]);
    //    }
    //    printf("]\n");
    //
    //    printf("\n--- gpu_edgeList ---\n[");
    //    for (int i = 0; i < *gpu_nEdges; i++) {
    //        if (i > 0) printf(", ");
    //        printf("%d", gpu_edgeList[i]);
    //    }
    //    printf("]\n");
    //
    //    printf("\n\n--- cpu_edgeId ---\n[");
    //    for (int i = 0; i < cpu_nEdges; i++) {
    //        if (i > 0) printf(", ");
    //        printf("%d", cpu_edgeId[i]);
    //    }
    //    printf("]\n");
    //
    //    printf("\n--- cpu_edgeList ---\n[");
    //    for (int i = 0; i < cpu_nEdges; i++) {
    //        if (i > 0) printf(", ");
    //        printf("%d", cpu_edgeList[i]);
    //    }
    //    printf("]\n\n");
    //
    //    printf("\n--- pfrh_neighbours ---\n[");
    //    for (int i = 0; i < nRH*8; i++) {
    //        if (i > 0) printf(", ");
    //        printf("%d", pfrh_neighbours[i]);
    //    }
    //    printf("]\n\n");

    if (*gpu_nEdges != cpu_nEdges) {
      printf("Different values of nEdges!\nGPU: %d\t\tCPU: %d\n", *gpu_nEdges, cpu_nEdges);
    } else {
      int mismatches_edgeId = 0, mismatches_edgeList = 0;
      for (int i = 0; i < cpu_nEdges; i++) {
        if (cpu_edgeId[i] != gpu_edgeId[i]) {
          printf("\tDifference in edgeId position %d:\tGPU: %d\t\tCPU: %d\n", i, gpu_edgeId[i], cpu_edgeId[i]);
          mismatches_edgeId++;
        }
        if (cpu_edgeList[i] != gpu_edgeList[i]) {
          printf("\tDifference in edgeList position %d:\tGPU: %d\t\tCPU: %d\n", i, gpu_edgeList[i], cpu_edgeList[i]);
          mismatches_edgeList++;
        }
      }

      if (mismatches_edgeId > 0)
        printf("Mismatches in edgeId: %d\n", mismatches_edgeId);
      else
        printf("All values in edgeId match for GPU & CPU\n");
      if (mismatches_edgeList > 0)
        printf("Mismatches in edgeList: %d\n", mismatches_edgeList);
      else
        printf("All values in edgeList match for GPU & CPU\n");
      printf("\n");
    }

    //    for (int i = 0; i < (int)nRH; i++) {
    //        int4 gpu4N = make_int4(pfrh_neighbours[i*8], pfrh_neighbours[i*8+1], pfrh_neighbours[i*8+2], pfrh_neighbours[i*8+3]);
    //        for (int n = 0; n < 4; n++) {
    //            if (cpu
    //        }
    //    }
  }

  __global__ void printRhfIndex(int* pfrh_topoId, int* topoRHCount, int* seedFracOffsets, int* pcrhfracind) {
    int seedIdx = 500;
    int offset = seedFracOffsets[seedIdx];
    int topoId = pfrh_topoId[seedIdx];
    int nRHF = topoRHCount[topoId];
    if (offset > -1 && topoId > -1) {
      printf(
          "seed %d has topoId %d and offset %d with %d expected rechit fractions:\n[", seedIdx, topoId, offset, nRHF);
      for (int r = offset; r < (offset + nRHF); r++) {
        if (r != offset)
          printf(", ");
        printf("%d", pcrhfracind[r]);
      }
      printf("]\n\n");
    }
  }

  void PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream_t cudaStream,
      int nEdges,
      ::hcal::PFRecHitCollection<::pf::common::DevStoragePolicy> const& inputPFRecHits,
      ::PFClustering::HCAL::InputDataGPU& inputGPU,
      ::PFClustering::HCAL::OutputDataCPU& outputCPU,
      ::PFClustering::HCAL::OutputDataGPU& outputGPU,
      ::PFClustering::HCAL::ScratchDataGPU& scratchGPU,
      float (&timer)[8]) {

#ifdef DEBUG_GPU_HCAL
    cudaProfilerStart();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, cudaStream);
#endif

    int nRH = inputPFRecHits.size;

    // Combined seeding & topo clustering thresholds, array initialization

    seedingTopoThreshKernel_HCAL<<<(nRH + 31) / 32, 64, 0, cudaStream>>>(nRH,
                                                                         inputPFRecHits.pfrh_energy.get(),
                                                                         inputPFRecHits.pfrh_x.get(),
                                                                         inputPFRecHits.pfrh_y.get(),
                                                                         inputPFRecHits.pfrh_z.get(),
                                                                         outputGPU.pfrh_isSeed.get(),
                                                                         outputGPU.pfrh_topoId.get(),
                                                                         outputGPU.pfrh_passTopoThresh.get(),
                                                                         inputPFRecHits.pfrh_layer.get(),
                                                                         inputPFRecHits.pfrh_depth.get(),
                                                                         inputPFRecHits.pfrh_neighbours.get(),
                                                                         scratchGPU.rhcount.get(),
                                                                         outputGPU.topoSeedCount.get(),
                                                                         outputGPU.topoRHCount.get(),
                                                                         outputGPU.seedFracOffsets.get(),
                                                                         outputGPU.topoSeedOffsets.get(),
                                                                         outputGPU.topoSeedList.get(),
                                                                         outputGPU.pfc_iter.get());

    cudaCheck(cudaStreamSynchronize(cudaStream));

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaEventRecord(start, cudaStream);
#endif

       // prepareTopoInputsSerial<<<1, 1, 4 * (8+4) * sizeof(int), cudaStream>>>(
       //     nRH,
       //     outputGPU.nEdges.get(),
       //     outputGPU.pfrh_passTopoThresh.get(),
       //     inputPFRecHits.pfrh_neighbours.get(),
       //     scratchGPU.pfrh_edgeId.get(),
       //     scratchGPU.pfrh_edgeList.get());

    // Topo clustering
    // Fill edgeId, edgeList arrays with rechit neighbors
    // Has a bug when using more than 128 threads..
    prepareTopoInputs<<<1, 128, 128 * (8 + 4) * sizeof(int), cudaStream>>>(nRH,
                                                                           outputGPU.nEdges.get(),
                                                                           outputGPU.pfrh_passTopoThresh.get(),
                                                                           inputPFRecHits.pfrh_neighbours.get(),
                                                                           scratchGPU.pfrh_edgeId.get(),
                                                                           scratchGPU.pfrh_edgeList.get());
    cudaCheck(cudaStreamSynchronize(cudaStream));

    //    prepareTopoInputs<<<1, 256, 256 * (8+4) * sizeof(int), cudaStream>>>(
    //        nRH,
    //        outputGPU.nEdges.get(),
    //        outputGPU.pfrh_passTopoThresh.get(),
    //        inputPFRecHits.pfrh_neighbours.get(),
    //        scratchGPU.pfrh_edgeId.get(),
    //        scratchGPU.pfrh_edgeList.get());

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[4], start, stop);
    //printf("\nprepareTopoInputs took %f ms\n", timer[4]);

    compareEdgeArrays<<<1, 1, 0, cudaStream>>>(outputGPU.nEdges.get(),
                                               scratchGPU.pfrh_edgeId.get(),
                                               scratchGPU.pfrh_edgeList.get(),
                                               nEdges,
                                               inputGPU.pfrh_edgeId.get(),
                                               inputGPU.pfrh_edgeList.get(),
                                               nRH,
                                               inputGPU.pfNeighFourInd.get(),
                                               inputPFRecHits.pfrh_neighbours.get());

    cudaEventRecord(start, cudaStream);
#endif

    // Topo clustering
    //topoClusterLinking<<<1, 512, 0, cudaStream>>>(nRH,
    topoClusterLinkingKH<<<1, 512, 0, cudaStream>>>(nRH,
						    outputGPU.nEdges.get(),
						    //inputPFRecHits.pfrh_energy.get(), // temporary entry for debugging
						    outputGPU.pfrh_topoId.get(),
						    scratchGPU.pfrh_edgeId.get(),
						    scratchGPU.pfrh_edgeList.get(),
						    scratchGPU.pfrh_edgeMask.get(),
						    //inputGPU.pfrh_edgeMask.get(),
						    outputGPU.pfrh_passTopoThresh.get(),
						    outputGPU.topoIter.get());
    cudaCheck(cudaStreamSynchronize(cudaStream));

    topoClusterContraction<<<1, 512, 0, cudaStream>>>(nRH,
                                                      outputGPU.pfrh_topoId.get(),
                                                      outputGPU.pfrh_isSeed.get(),
                                                      //inputPFRecHits.pfrh_neighbours.get(), // temporary entry for debugging
                                                      scratchGPU.rhcount.get(),
                                                      outputGPU.topoSeedCount.get(),
                                                      outputGPU.topoRHCount.get(),
                                                      outputGPU.seedFracOffsets.get(),
                                                      outputGPU.topoSeedOffsets.get(),
                                                      outputGPU.topoSeedList.get(),
                                                      outputGPU.pcrh_fracInd.get(),
                                                      outputGPU.pcrh_frac.get(),
                                                      outputGPU.pcrhFracSize.get());

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaEventRecord(start, cudaStream);
#endif

    dim3 grid((nRH + 31) / 32, (nRH + 31) / 32);
    dim3 block(32, 32);

    fillRhfIndex<<<grid, block, 0, cudaStream>>>(nRH,
                                                 outputGPU.pfrh_topoId.get(),
                                                 outputGPU.pfrh_isSeed.get(),
                                                 outputGPU.topoSeedCount.get(),
                                                 outputGPU.topoRHCount.get(),
                                                 outputGPU.seedFracOffsets.get(),
                                                 scratchGPU.rhcount.get(),
                                                 outputGPU.pcrh_fracInd.get());

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaEventRecord(start, cudaStream);
#endif

    hcalFastCluster_selection<<<nRH, 256, 0, cudaStream>>>(nRH,
                                                           inputPFRecHits.pfrh_x.get(),
                                                           inputPFRecHits.pfrh_y.get(),
                                                           inputPFRecHits.pfrh_z.get(),
                                                           inputPFRecHits.pfrh_energy.get(),
                                                           outputGPU.pfrh_topoId.get(),
                                                           outputGPU.pfrh_isSeed.get(),
                                                           inputPFRecHits.pfrh_layer.get(),
                                                           inputPFRecHits.pfrh_depth.get(),
                                                           inputPFRecHits.pfrh_neighbours.get(),
                                                           outputGPU.pcrh_frac.get(),
                                                           outputGPU.pcrh_fracInd.get(),
                                                           inputGPU.pcrh_fracSum.get(),
                                                           scratchGPU.rhcount.get(),
                                                           outputGPU.topoSeedCount.get(),
                                                           outputGPU.topoRHCount.get(),
                                                           outputGPU.seedFracOffsets.get(),
                                                           outputGPU.topoSeedOffsets.get(),
                                                           outputGPU.topoSeedList.get(),
                                                           inputGPU.pfc_pos4.get(),
                                                           inputGPU.pfc_prevPos4.get(),
                                                           inputGPU.pfc_energy.get(),
                                                           outputGPU.pfc_iter.get());
#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[3], start, stop);
    cudaProfilerStop();
#endif
  }
}  // namespace PFClusterCudaHCAL

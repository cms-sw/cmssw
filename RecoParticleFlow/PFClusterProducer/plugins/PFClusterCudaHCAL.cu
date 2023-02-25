#include <cmath>
#include <iostream>
#include <set>

// CUDA include files
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Eigen include files
#include <Eigen/Dense>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "PFClusterCudaHCAL.h"

using PFClustering::common::PFLayer;

constexpr const float PI_F = 3.141592654f;

// Number of neighbors considered for topo clustering

namespace PFClusterCudaHCAL {
  //
  // --- kernel summary --
  // PFRechitToPFCluster_HCAL_entryPoint
  //   seedingTopoThreshKernel_HCAL: apply seeding/topo-clustering threshold to RecHits, also ensure a peak (outputs: pfrh_isSeed, pfrh_passTopoThresh) [OutputDataGPU]
  //   prepareTopoInputs: prepare "edge" data (outputs: nEdges, pfrh_edgeId, pfrh_edgeList [nEdges dimension])
  //   ECLCC: run topo clustering (output: pfrh_topoId)
  //   topoClusterContraction: find parent of parent (or parent (of parent ...)) (outputs: pfrh_topoId, topoSeedCount, topoSeedOffsets, topoSeedList, seedFracOffsets, pcrhfracind, pcrhfrac)
  //   fillRhfIndex: fill rhfracind (PFCluster RecHitFraction constituent PFRecHit indices)
  //   hcalFastCluster_selection
  //     dev_hcalFastCluster_optimizedSimple
  //     dev_hcalFastCluster_optimizedComplex
  //     dev_hcalFastCluster_original
  // [aux]
  //     sortEight
  //     sortSwap
  // [not used]
  //   [compareEdgeArrays] used only for debugging
  //   compareEdgeArrays

  /*
  ECL-CC code: ECL-CC is a connected components graph algorithm. The CUDA
  implementation thereof is quite fast. It operates on graphs stored in
  binary CSR format.

  Copyright (c) 2017-2020, Texas State University. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Texas State University nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  Authors: Jayadharini Jaiganesh and Martin Burtscher

  URL: The latest version of this code is available at
  https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/.

  Publication: This work is described in detail in the following paper.
  Jayadharini Jaiganesh and Martin Burtscher. A High-Performance Connected
  Components Implementation for GPUs. Proceedings of the 2018 ACM International
  Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
  June 2018.
  */

  static const int ThreadsPerBlock = 256;
  static const int warpsize = 32;

  /* initialize with first smaller neighbor ID */

  __global__ void ECLCC_init(const int nodes,
                             const int* const __restrict__ nidx,
                             const int* const __restrict__ nlist,
                             int* const __restrict__ nstat,
                             int* topL,
                             int* posL,
                             int* topH,
                             int* posH) {
    const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int incr = gridDim.x * ThreadsPerBlock;

    for (int v = from; v < nodes; v += incr) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int m = v;
      int i = beg;
      while ((m == v) && (i < end)) {
        m = min(m, nlist[i]);
        i++;
      }
      nstat[v] = m;
    }

    if (from == 0) {
      *topL = 0;
      *posL = 0;
      *topH = nodes - 1;
      *posH = nodes - 1;
    }
  }

  /* intermediate pointer jumping */

  __device__ int representative(const int idx, int* const __restrict__ nstat) {
    int curr = nstat[idx];
    if (curr != idx) {
      int next, prev = idx;
      while (curr > (next = nstat[curr])) {
        nstat[prev] = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  /* process low-degree vertices at thread granularity and fill worklists */

  __global__ void ECLCC_compute1(const int nodes,
                                 const int* const __restrict__ nidx,
                                 const int* const __restrict__ nlist,
                                 int* const __restrict__ nstat,
                                 int* const __restrict__ wl,
                                 int* topL,
                                 int* topH) {
    const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int incr = gridDim.x * ThreadsPerBlock;

    for (int v = from; v < nodes; v += incr) {
      const int vstat = nstat[v];
      if (v != vstat) {
        const int beg = nidx[v];
        const int end = nidx[v + 1];
        int deg = end - beg;
        if (deg > 16) {
          int idx;
          if (deg <= 352) {
            idx = atomicAdd(&*topL, 1);
          } else {
            idx = atomicAdd(&*topH, -1);
          }
          wl[idx] = v;
        } else {
          int vstat = representative(v, nstat);
          for (int i = beg; i < end; i++) {
            const int nli = nlist[i];
            if (v > nli) {
              int ostat = representative(nli, nstat);
              bool repeat;
              do {
                repeat = false;
                if (vstat != ostat) {
                  int ret;
                  if (vstat < ostat) {
                    if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                      ostat = ret;
                      repeat = true;
                    }
                  } else {
                    if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                      vstat = ret;
                      repeat = true;
                    }
                  }
                }
              } while (repeat);
            }
          }
        }
      }
    }
  }

  /* process medium-degree vertices at warp granularity */

  __global__ void ECLCC_compute2(const int nodes,
                                 const int* const __restrict__ nidx,
                                 const int* const __restrict__ nlist,
                                 int* const __restrict__ nstat,
                                 const int* const __restrict__ wl,
                                 int* topL,
                                 int* posL) {
    const int lane = threadIdx.x % warpsize;

    int idx;
    if (lane == 0)
      idx = atomicAdd(*&posL, 1);
    idx = __shfl_sync(0xffffffff, idx, 0);
    while (idx < *topL) {
      const int v = wl[idx];
      int vstat = representative(v, nstat);
      for (int i = nidx[v] + lane; i < nidx[v + 1]; i += warpsize) {
        const int nli = nlist[i];
        if (v > nli) {
          int ostat = representative(nli, nstat);
          bool repeat;
          do {
            repeat = false;
            if (vstat != ostat) {
              int ret;
              if (vstat < ostat) {
                if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                  ostat = ret;
                  repeat = true;
                }
              } else {
                if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                  vstat = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
      }
      if (lane == 0)
        idx = atomicAdd(*&posL, 1);
      idx = __shfl_sync(0xffffffff, idx, 0);
    }
  }

  /* process high-degree vertices at block granularity */

  __global__ void ECLCC_compute3(const int nodes,
                                 const int* const __restrict__ nidx,
                                 const int* const __restrict__ nlist,
                                 int* const __restrict__ nstat,
                                 const int* const __restrict__ wl,
                                 int* topH,
                                 int* posH) {
    __shared__ int vB;
    if (threadIdx.x == 0) {
      const int idx = atomicAdd(&*posH, -1);
      vB = (idx > *topH) ? wl[idx] : -1;
    }
    __syncthreads();
    while (vB >= 0) {
      const int v = vB;
      __syncthreads();
      int vstat = representative(v, nstat);
      for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ThreadsPerBlock) {
        const int nli = nlist[i];
        if (v > nli) {
          int ostat = representative(nli, nstat);
          bool repeat;
          do {
            repeat = false;
            if (vstat != ostat) {
              int ret;
              if (vstat < ostat) {
                if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                  ostat = ret;
                  repeat = true;
                }
              } else {
                if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                  vstat = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        const int idx = atomicAdd(*&posH, -1);
        vB = (idx > *topH) ? wl[idx] : -1;
      }
      __syncthreads();
    }
  }

  /* link all vertices to sink */

  __global__ void ECLCC_flatten(const int nodes,
                                const int* const __restrict__ nidx,
                                const int* const __restrict__ nlist,
                                int* const __restrict__ nstat) {
    const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int incr = gridDim.x * ThreadsPerBlock;

    for (int v = from; v < nodes; v += incr) {
      int next, vstat = nstat[v];
      const int old = vstat;
      while (vstat > (next = nstat[vstat])) {
        vstat = next;
      }
      if (old != vstat)
        nstat[v] = vstat;
    }
  }
  //
  // ECL-CC ends
  //

  __device__ __forceinline__ float timeResolution2Endcap(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                                         const float energy) {
    float res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < pfClusParams.endcapTimeResConsts_threshLowE()) {
      if (pfClusParams.endcapTimeResConsts_corrTermLowE() > 0.) {  // different parametrisation
        const float res = pfClusParams.endcapTimeResConsts_noiseTermLowE() / energy +
                          pfClusParams.endcapTimeResConsts_corrTermLowE() / (energy * energy);
        res2 = res * res;
      } else {
        const float noiseDivE = pfClusParams.endcapTimeResConsts_noiseTermLowE() / energy;
        res2 = noiseDivE * noiseDivE + pfClusParams.endcapTimeResConsts_constantTermLowE2();
      }
    } else if (energy < pfClusParams.endcapTimeResConsts_threshHighE()) {
      const float noiseDivE = pfClusParams.endcapTimeResConsts_noiseTerm() / energy;
      res2 = noiseDivE * noiseDivE + pfClusParams.endcapTimeResConsts_constantTerm2();
    } else  // if (energy >=threshHighE_)
      res2 = pfClusParams.endcapTimeResConsts_resHighE2();

    if (res2 > 10000.)
      return 10000.;
    return res2;
  }

  __device__ __forceinline__ float timeResolution2Barrel(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                                         const float energy) {
    float res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < pfClusParams.barrelTimeResConsts_threshLowE()) {
      if (pfClusParams.barrelTimeResConsts_corrTermLowE() > 0.) {  // different parametrisation
        const float res = pfClusParams.barrelTimeResConsts_noiseTermLowE() / energy +
                          pfClusParams.barrelTimeResConsts_corrTermLowE() / (energy * energy);
        res2 = res * res;
      } else {
        const float noiseDivE = pfClusParams.barrelTimeResConsts_noiseTermLowE() / energy;
        res2 = noiseDivE * noiseDivE + pfClusParams.barrelTimeResConsts_constantTermLowE2();
      }
    } else if (energy < pfClusParams.barrelTimeResConsts_threshHighE()) {
      const float noiseDivE = pfClusParams.barrelTimeResConsts_noiseTerm() / energy;
      res2 = noiseDivE * noiseDivE + pfClusParams.barrelTimeResConsts_constantTerm2();
    } else  // if (energy >=threshHighE_)
      res2 = pfClusParams.barrelTimeResConsts_resHighE2();

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

  __global__ void seedingTopoThreshKernel_HCAL(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                               size_t size,
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
      if ((layer == PFLayer::HCAL_BARREL1 && energy > pfClusParams.seedEThresholdEB_vec()[depthOffset] &&
           pt2 > pfClusParams.seedPt2ThresholdEB()) ||
          (layer == PFLayer::HCAL_ENDCAP && energy > pfClusParams.seedEThresholdEE_vec()[depthOffset] &&
           pt2 > pfClusParams.seedPt2ThresholdEE())) {
        pfrh_isSeed[i] = 1;
        for (int k = 0; k < 4; k++) {  // Does this seed candidate have a higher energy than four neighbours
          if (neigh4_Ind[8 * i + k] < 0)
            continue;
          if (energy < pfrh_energy[neigh4_Ind[8 * i + k]]) {
            pfrh_isSeed[i] = 0;
            //pfrh_topoId[i]=-1;
            break;
          }
        }
        //         for(int k=0; k<pfClusParams.nNeigh(); k++){
        //           if(neigh4_Ind[pfClusParams.nNeigh()*i+k]<0) continue;
        //           if(energy < pfrh_energy[neigh4_Ind[pfClusParams.nNeigh()*i+k]]){
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
      if ((layer == PFLayer::HCAL_ENDCAP && energy > pfClusParams.topoEThresholdEE_vec()[depthOffset]) ||
          (layer == PFLayer::HCAL_BARREL1 && energy > pfClusParams.topoEThresholdEB_vec()[depthOffset])) {
        pfrh_passTopoThresh[i] = true;
      }
      //else { pfrh_passTopoThresh[i] = false; }
      else {
        pfrh_passTopoThresh[i] = false;
        pfrh_topoId[i] = -1;
      }
    }
  }

  __device__ void dev_hcalFastCluster_optimizedSimple(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                                      int topoId,
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
      tol = pfClusParams.stoppingTolerance();  // stopping tolerance * tolerance scaling

      if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfrh_depth[i] - 1];
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

        d2 = dist2 / pfClusParams.showerSigma2();
        fraction = clusterEnergy * rhENormInv * expf(-0.5 * d2);

        // For single seed clusters, rechit fraction is either 1 (100%) or -1 (not included)
        if (fraction > pfClusParams.minFracTot() && d2 < 100.)
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
        //computeClusterPos(pfClusParams, clusterPos, rechitPos, rhEnergy, rhENormInv, debug);
        atomicAdd(&clusterPos.x, rhPos.x * rhPosNorm);
        atomicAdd(&clusterPos.y, rhPos.y * rhPosNorm);
        atomicAdd(&clusterPos.z, rhPos.z * rhPosNorm);
        atomicAdd(&clusterPos.w, rhPosNorm);  // position_norm
      }
      __syncthreads();

      if (tid == 0) {
        // Normalize the seed postiion
        if (clusterPos.w >= pfClusParams.minAllowedNormalization()) {
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
                   pfClusParams.minAllowedNormalization());
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
        notDone = (diff > tol) && (iter < pfClusParams.maxIterations());
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

  __device__ void dev_hcalFastCluster_optimizedComplex(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                                       int topoId,
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
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
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
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfrh_depth[i] - 1];
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

    auto computeClusterPos = [&](PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                 float4& pos4,
                                 float frac,
                                 int rhInd,
                                 bool isDebug) {
      float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
      const auto rh_energy = pfrh_energy[rhInd] * frac;
      const auto norm = (frac < pfClusParams.minFracInCalc() ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
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
      computeClusterPos(pfClusParams, seedInitClusterPos, 1., seedThreadIdx, debug);
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

          float d2 = dist2 / pfClusParams.showerSigma2();
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

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            //float fracpct = pcrhfrac[seedFracOffsets[i]+tid+1] / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > pfClusParams.minFracToKeep())) {
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
              computeClusterPos(pfClusParams, clusterPos[tid], frac, j, debug);
          }
        }
      }
      __syncthreads();

      // Position normalization
      if (tid < nSeeds) {
        if (clusterPos[tid].w >= pfClusParams.minAllowedNormalization()) {
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
                   pfClusParams.minAllowedNormalization());
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
        notDone = (diff > tol) && (iter < pfClusParams.maxIterations());
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
  __device__ void dev_hcalFastCluster_original(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                               int topoId,
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
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
      gridStride = blockDim.x;
      iter = 0;
      notDone = true;
      debug = false;

      int i = topoSeedList[topoSeedBegin];
      if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfrh_depth[i] - 1];
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

    auto computeClusterPos = [&](PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                 float4& pos4,
                                 float frac,
                                 int rhInd,
                                 bool isDebug) {
      float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
      const auto rh_energy = pfrh_energy[rhInd] * frac;
      const auto norm = (frac < pfClusParams.minFracInCalc() ? 0.0f : max(0.0f, logf(rh_energy * rhENormInv)));
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

          float d2 = dist2 / pfClusParams.showerSigma2();
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

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            //float fracpct = pcrhfrac[seedFracOffsets[i]+tid+1] / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > pfClusParams.minFracToKeep())) {
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
              computeClusterPos(pfClusParams, clusterPos[s], frac, j, debug);
          }
        }
      }
      __syncthreads();

      // Position normalization
      for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        if (clusterPos[s].w >= pfClusParams.minAllowedNormalization()) {
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
                   pfClusParams.minAllowedNormalization());
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
        notDone = (diff > tol) && (iter < pfClusParams.maxIterations());
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

  __global__ void hcalFastCluster_selection(PFClusteringParamsGPU::DeviceProduct::ConstView pfClusParams,
                                            size_t nRH,
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
                                            int* topoIds,
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
      topoId = topoIds[blockIdx.x];
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
        dev_hcalFastCluster_optimizedSimple(pfClusParams,
                                            topoId,
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
        dev_hcalFastCluster_optimizedComplex(pfClusParams,
                                             topoId,
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
        dev_hcalFastCluster_original(pfClusParams,
                                     topoId,
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
                                         int* pcrhFracSize,
                                         int* nRHFracs,
                                         int* nSeeds,
                                         int* nTopos,
                                         int* topoIds) {
    __shared__ int totalSeedOffset, totalSeedFracOffset;
    if (threadIdx.x == 0) {
      *nTopos = 0;
      *nSeeds = 0;
      *nRHFracs = 0;
      totalSeedOffset = 0;
      totalSeedFracOffset = 0;
      *pcrhFracSize = 0;
    }
    __syncthreads();
    // Now determine the number of seeds and rechits in each topo cluster
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
      int topoId = pfrh_parent[rhIdx];
      if (topoId > -1) {
        // Valid topo cluster
        atomicAdd(&topoRHCount[topoId], 1);
        // Valid topoId not counted yet
        if (topoId == rhIdx) {  // For every topo cluster, there is one rechit that meets this condition.
          int topoIdx = atomicAdd(&*nTopos, 1);
          topoIds[topoIdx] = topoId;  // topoId: the smallest index of rechits that belong to a topo cluster.
        }
        // This is a cluster seed
        if (pfrh_isSeed[rhIdx]) {  // # of seeds in this topo cluster
          atomicAdd(&topoSeedCount[topoId], 1);
          atomicAdd(&*nSeeds, 1);
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
      *nRHFracs = totalSeedFracOffset;
      if (*pcrhFracSize > 200000)  // Warning in case the fraction is too large
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
      int topoId = pfrh_parent[i];
      if (topoId == pfrh_parent[j] && topoId > -1 && pfrh_isSeed[i] && !pfrh_isSeed[j]) {
        int k = atomicAdd(&rhCount[i], 1);        // Increment the number of rechit fractions for this seed
        pcrhfracind[seedFracOffsets[i] + k] = j;  // Save this rechit index
      }
    }
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
    assert(size == 32);
    int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (int offset = 1; offset < size; offset <<= 1) {
      int t = s_Data[pos] + s_Data[pos - offset];
      __syncwarp();
      s_Data[pos] = t;
      __syncwarp();
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

  __global__ void prepareTopoInputs(int nRH,
                                    int* nEdges,
                                    const int* pfrh_passTopoThresh,
                                    const int* pfrh_neighbours,
                                    int* pfrh_edgeIdx,
                                    int* pfrh_edgeList,
                                    int* pfrh_parent) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      *nEdges = nRH * 8;
      pfrh_edgeIdx[nRH] = nRH * 8;
    }

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nRH; i += blockDim.x * gridDim.x) {
      pfrh_edgeIdx[i] = i * 8;
      pfrh_parent[i] = 0;
      for (int j = 0; j < 8; j++) {
        if (pfrh_neighbours[i * 8 + j] == -1)
          pfrh_edgeList[i * 8 + j] = i;
        else
          pfrh_edgeList[i * 8 + j] = pfrh_neighbours[i * 8 + j];
      }
    }

    return;
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

  void PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream_t cudaStream,
      PFClusteringParamsGPU::DeviceProduct const& pfClusParams,
      ::hcal::PFRecHitCollection<::pf::common::DevStoragePolicy> const& inputPFRecHits,
      ::PFClustering::HCAL::OutputPFClusterDataGPU& outputGPU2,
      ::PFClustering::HCAL::OutputDataGPU& outputGPU,
      ::PFClustering::HCAL::ScratchDataGPU& scratchGPU,
      float (&timer)[8]) {
    const int threadsPerBlock = 256;
    const int nRH = inputPFRecHits.size;
    const int blocks = (nRH + threadsPerBlock - 1) / threadsPerBlock;

    outputGPU2.allocate(nRH, cudaStream);
    outputGPU2.allocate_rhfrac(nRH, cudaStream);

    // Combined seeding & topo clustering thresholds, array initialization
    seedingTopoThreshKernel_HCAL<<<(nRH + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, 0, cudaStream>>>(
        pfClusParams.const_view(),
        nRH,
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

    // Topo clustering
    // Fill edgeId, edgeList arrays with rechit neighbors
    prepareTopoInputs<<<blocks, threadsPerBlock, 0, cudaStream>>>(nRH,
                                                                  outputGPU.nEdges.get(),
                                                                  outputGPU.pfrh_passTopoThresh.get(),
                                                                  inputPFRecHits.pfrh_neighbours.get(),
                                                                  scratchGPU.pfrh_edgeIdx.get(),
                                                                  scratchGPU.pfrh_edgeList.get(),
                                                                  outputGPU.pfrh_topoId.get());
    // Topo clustering
    ECLCC_init<<<blocks, threadsPerBlock, 0, cudaStream>>>(nRH,
                                                           scratchGPU.pfrh_edgeIdx.get(),
                                                           scratchGPU.pfrh_edgeList.get(),
                                                           outputGPU.pfrh_topoId.get(),
                                                           scratchGPU.topL.get(),
                                                           scratchGPU.posL.get(),
                                                           scratchGPU.topH.get(),
                                                           scratchGPU.posH.get());
    ECLCC_compute1<<<blocks, threadsPerBlock, 0, cudaStream>>>(nRH,
                                                               scratchGPU.pfrh_edgeIdx.get(),
                                                               scratchGPU.pfrh_edgeList.get(),
                                                               outputGPU.pfrh_topoId.get(),
                                                               scratchGPU.wl_d.get(),
                                                               scratchGPU.topL.get(),
                                                               scratchGPU.topH.get());
    ECLCC_compute2<<<blocks, threadsPerBlock, 0, cudaStream>>>(nRH,
                                                               scratchGPU.pfrh_edgeIdx.get(),
                                                               scratchGPU.pfrh_edgeList.get(),
                                                               outputGPU.pfrh_topoId.get(),
                                                               scratchGPU.wl_d.get(),
                                                               scratchGPU.topL.get(),
                                                               scratchGPU.posL.get());
    ECLCC_compute3<<<blocks, threadsPerBlock, 0, cudaStream>>>(nRH,
                                                               scratchGPU.pfrh_edgeIdx.get(),
                                                               scratchGPU.pfrh_edgeList.get(),
                                                               outputGPU.pfrh_topoId.get(),
                                                               scratchGPU.wl_d.get(),
                                                               scratchGPU.topH.get(),
                                                               scratchGPU.posH.get());
    ECLCC_flatten<<<blocks, threadsPerBlock, 0, cudaStream>>>(
        nRH, scratchGPU.pfrh_edgeIdx.get(), scratchGPU.pfrh_edgeList.get(), outputGPU.pfrh_topoId.get());

    topoClusterContraction<<<1, 512, 0, cudaStream>>>(nRH,
                                                      outputGPU.pfrh_topoId.get(),
                                                      outputGPU.pfrh_isSeed.get(),
                                                      scratchGPU.rhcount.get(),
                                                      outputGPU.topoSeedCount.get(),
                                                      outputGPU.topoRHCount.get(),
                                                      outputGPU.seedFracOffsets.get(),
                                                      outputGPU.topoSeedOffsets.get(),
                                                      outputGPU.topoSeedList.get(),
                                                      outputGPU.pcrh_fracInd.get(),
                                                      outputGPU.pcrh_frac.get(),
                                                      outputGPU.pcrhFracSize.get(),
                                                      scratchGPU.nRHFracs.get(),
                                                      scratchGPU.nSeeds.get(),
                                                      scratchGPU.nTopos.get(),
                                                      scratchGPU.topoIds.get());

    dim3 grid((nRH + 31) / 32, (nRH + 31) / 32);
    dim3 block(32, 32);

    int nTopos_h;
    cudaCheck(cudaMemcpyAsync(&nTopos_h, scratchGPU.nTopos.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));

    // x: seeds, y: non-seeds
    fillRhfIndex<<<grid, block, 0, cudaStream>>>(nRH,
                                                 outputGPU.pfrh_topoId.get(),
                                                 outputGPU.pfrh_isSeed.get(),
                                                 outputGPU.topoSeedCount.get(),
                                                 outputGPU.topoRHCount.get(),
                                                 outputGPU.seedFracOffsets.get(),
                                                 scratchGPU.rhcount.get(),
                                                 outputGPU.pcrh_fracInd.get());

    // grid -> topo cluster
    // thread -> pfrechits in each topo cluster
    hcalFastCluster_selection<<<nTopos_h, threadsPerBlock, 0, cudaStream>>>(pfClusParams.const_view(),
                                                                            nRH,
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
                                                                            scratchGPU.topoIds.get(),
                                                                            scratchGPU.pcrh_fracSum.get(),
                                                                            scratchGPU.rhcount.get(),
                                                                            outputGPU.topoSeedCount.get(),
                                                                            outputGPU.topoRHCount.get(),
                                                                            outputGPU.seedFracOffsets.get(),
                                                                            outputGPU.topoSeedOffsets.get(),
                                                                            outputGPU.topoSeedList.get(),
                                                                            outputGPU.pfc_pos4.get(),
                                                                            scratchGPU.pfc_prevPos4.get(),
                                                                            outputGPU.pfc_energy.get(),
                                                                            outputGPU.pfc_iter.get());
  }
}  // namespace PFClusterCudaHCAL

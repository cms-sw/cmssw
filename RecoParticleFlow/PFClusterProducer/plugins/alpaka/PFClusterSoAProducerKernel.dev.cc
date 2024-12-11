#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "FWCore/Utilities/interface/bit_cast.h"
#include "HeterogeneousCore/AlpakaInterface/interface/atomicMaxF.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterECLCC.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterSoAProducerKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  using namespace reco::pfClustering;

  static constexpr int threadsPerBlockForClustering = 512;
  static constexpr uint32_t blocksForExoticClusters = 4;

  // cutoffFraction -> Is a rechit almost entirely attributed to one cluster
  // cutoffDistance -> Is a rechit close enough to a cluster to be associated
  // Values are from RecoParticleFlow/PFClusterProducer/plugins/Basic2DGenericPFlowClusterizer.cc
  static constexpr float cutoffDistance = 100.;
  static constexpr float cutoffFraction = 0.9999;

  static constexpr uint32_t kHBHalf = 1296;
  static constexpr uint32_t maxTopoInput = 2 * kHBHalf;

  // Calculation of dR2 for Clustering
  ALPAKA_FN_ACC ALPAKA_FN_INLINE static float dR2(Position4 pos1, Position4 pos2) {
    float mag1 = sqrtf(pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
    float cosTheta1 = mag1 > 0.0 ? pos1.z / mag1 : 1.0;
    float eta1 = 0.5f * logf((1.0f + cosTheta1) / (1.0f - cosTheta1));
    float phi1 = atan2f(pos1.y, pos1.x);

    float mag2 = sqrtf(pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
    float cosTheta2 = mag2 > 0.0 ? pos2.z / mag2 : 1.0;
    float eta2 = 0.5f * logf((1.0f + cosTheta2) / (1.0f - cosTheta2));
    float phi2 = atan2f(pos2.y, pos2.x);

    float deta = eta2 - eta1;
    constexpr const float fPI = M_PI;
    float dphi = std::abs(std::abs(phi2 - phi1) - fPI) - fPI;
    return (deta * deta + dphi * dphi);
  }

  // Get index of seed
  ALPAKA_FN_ACC static auto getSeedRhIdx(int* seeds, int seedNum) { return seeds[seedNum]; }

  // Get rechit fraction of a given rechit for a given seed
  ALPAKA_FN_ACC static auto getRhFrac(reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                      int topoSeedBegin,
                                      reco::PFRecHitFractionDeviceCollection::View fracView,
                                      int seedNum,
                                      int rhNum) {
    int seedIdx = pfClusteringVars[topoSeedBegin + seedNum].topoSeedList();
    return fracView[pfClusteringVars[seedIdx].seedFracOffsets() + rhNum].frac();
  }

  // Cluster position calculation
  template <bool debug = false>
  ALPAKA_FN_ACC static void updateClusterPos(::reco::PFClusterParamsSoA::ConstView pfClusParams,
                                             Position4& pos4,
                                             float frac,
                                             int rhInd,
                                             reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                             float rhENormInv) {
    Position4 rechitPos = Position4{pfRecHits[rhInd].x(), pfRecHits[rhInd].y(), pfRecHits[rhInd].z(), 1.0};
    const auto rh_energy = pfRecHits[rhInd].energy() * frac;
    const auto norm = (frac < pfClusParams.minFracInCalc() ? 0.0f : std::max(0.0f, logf(rh_energy * rhENormInv)));
    if constexpr (debug)
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
  }

  // Processing single seed clusters
  // Device function designed to be called by all threads of a given block
  template <bool debug = false>
  ALPAKA_FN_ACC static void hcalFastCluster_singleSeed(
      const Acc1D& acc,
      ::reco::PFClusterParamsSoA::ConstView pfClusParams,
      const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
      int topoId,   // from selection
      int nRHTopo,  // from selection
      reco::PFRecHitDeviceCollection::ConstView pfRecHits,
      reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
      reco::PFClusterDeviceCollection::View clusterView,
      reco::PFRecHitFractionDeviceCollection::View fracView) {
    int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // thread index is rechit number
    // Declaration of shared variables
    int& i = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& nRHOther = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    unsigned int& iter = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& clusterEnergy = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& seedEnergy = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    Position4& clusterPos = alpaka::declareSharedVar<Position4, __COUNTER__>(acc);
    Position4& prevClusterPos = alpaka::declareSharedVar<Position4, __COUNTER__>(acc);
    Position4& seedPos = alpaka::declareSharedVar<Position4, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    if (once_per_block(acc)) {
      i = pfClusteringVars[pfClusteringVars[topoId].topoSeedOffsets()].topoSeedList();  // i is the seed rechit index
      nRHOther = nRHTopo - 1;                                                           // number of non-seed rechits
      seedPos = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.};
      clusterPos = seedPos;  // Initial cluster position is just the seed
      prevClusterPos = seedPos;
      seedEnergy = pfRecHits[i].energy();
      clusterEnergy = seedEnergy;
      tol = pfClusParams.stoppingTolerance();  // stopping tolerance * tolerance scaling

      if (topology.cutsFromDB()) {
        rhENormInv = (1.f / topology[pfRecHits[i].denseId()].noiseThreshold());
      } else {
        if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
          rhENormInv = pfClusParams.recHitEnergyNormInvHB_vec()[pfRecHits[i].depth() - 1];
        else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
          rhENormInv = pfClusParams.recHitEnergyNormInvHE_vec()[pfRecHits[i].depth() - 1];
        else {
          rhENormInv = 0.;
          printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
        }
      }

      iter = 0;
      notDone = true;
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    int j = -1;  // j is the rechit index for this thread
    int rhFracOffset = -1;
    Position4 rhPos;
    float rhEnergy = -1., rhPosNorm = -1.;

    if (tid < nRHOther) {
      rhFracOffset =
          pfClusteringVars[i].seedFracOffsets() + tid + 1;  // Offset for this rechit in pcrhfrac, pcrhfracidx arrays
      j = fracView[rhFracOffset].pfrhIdx();                 // rechit index for this thread
      rhPos = Position4{pfRecHits[j].x(), pfRecHits[j].y(), pfRecHits[j].z(), 1.};
      rhEnergy = pfRecHits[j].energy();
      rhPosNorm = fmaxf(0., logf(rhEnergy * rhENormInv));
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    do {
      if constexpr (debug) {
        if (once_per_block(acc))
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }
      float dist2 = -1., d2 = -1., fraction = -1.;
      if (tid < nRHOther) {
        // Rechit distance calculation
        dist2 = (clusterPos.x - rhPos.x) * (clusterPos.x - rhPos.x) +
                (clusterPos.y - rhPos.y) * (clusterPos.y - rhPos.y) +
                (clusterPos.z - rhPos.z) * (clusterPos.z - rhPos.z);

        d2 = dist2 / pfClusParams.showerSigma2();
        fraction = clusterEnergy * rhENormInv * expf(-0.5f * d2);

        // For single seed clusters, rechit fraction is either 1 (100%) or -1 (not included)
        if (fraction > pfClusParams.minFracTot() && d2 < cutoffDistance)
          fraction = 1.;
        else
          fraction = -1.;
        fracView[rhFracOffset].frac() = fraction;
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if constexpr (debug) {
        if (once_per_block(acc))
          printf("Computing cluster position for topoId %d\n", topoId);
      }

      if (once_per_block(acc)) {
        // Reset cluster position and energy
        clusterPos = seedPos;
        clusterEnergy = seedEnergy;
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Recalculate cluster position and energy
      if (fraction > -0.5) {
        alpaka::atomicAdd(acc, &clusterEnergy, rhEnergy, alpaka::hierarchy::Threads{});
        alpaka::atomicAdd(acc, &clusterPos.x, rhPos.x * rhPosNorm, alpaka::hierarchy::Threads{});
        alpaka::atomicAdd(acc, &clusterPos.y, rhPos.y * rhPosNorm, alpaka::hierarchy::Threads{});
        alpaka::atomicAdd(acc, &clusterPos.z, rhPos.z * rhPosNorm, alpaka::hierarchy::Threads{});
        alpaka::atomicAdd(acc, &clusterPos.w, rhPosNorm, alpaka::hierarchy::Threads{});  // position_norm
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (once_per_block(acc)) {
        // Normalize the seed postiion
        if (clusterPos.w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos.x /= clusterPos.w;
          clusterPos.y /= clusterPos.w;
          clusterPos.z /= clusterPos.w;

          if constexpr (debug)
            printf("\tPF cluster (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   i,
                   clusterEnergy,
                   clusterPos.x,
                   clusterPos.y,
                   clusterPos.z);
        } else {
          if constexpr (debug)
            printf("\tPF cluster (seed %d) position norm (%f) less than minimum (%f)\n",
                   i,
                   clusterPos.w,
                   pfClusParams.minAllowedNormalization());
          clusterPos.x = 0.;
          clusterPos.y = 0.;
          clusterPos.z = 0.;
        }
        float diff2 = dR2(prevClusterPos, clusterPos);
        if constexpr (debug)
          printf("\tPF cluster (seed %d) has diff2 = %f\n", i, diff2);
        prevClusterPos = clusterPos;  // Save clusterPos

        float tol2 = tol * tol;
        iter++;
        notDone = (diff2 > tol2) && (iter < pfClusParams.maxIterations());
        if constexpr (debug) {
          if (diff2 > tol2)
            printf("\tTopoId %d has diff2 = %f greater than squared tolerance %f (continuing)\n", topoId, diff2, tol2);
          else if constexpr (debug)
            printf("\tTopoId %d has diff2 = %f LESS than squared tolerance %f (terminating!)\n", topoId, diff2, tol2);
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync
    } while (notDone);  // shared variable condition ensures synchronization is well defined
    if (once_per_block(acc)) {  // Cluster is finalized, assign cluster information to te SoA
      int rhIdx =
          pfClusteringVars[pfClusteringVars[topoId].topoSeedOffsets()].topoSeedList();  // i is the seed rechit index
      int seedIdx = pfClusteringVars[rhIdx].rhIdxToSeedIdx();
      clusterView[seedIdx].energy() = clusterEnergy;
      clusterView[seedIdx].x() = clusterPos.x;
      clusterView[seedIdx].y() = clusterPos.y;
      clusterView[seedIdx].z() = clusterPos.z;
    }
  }

  // Processing clusters up to 100 seeds and 512 non-seed rechits using shared memory accesses
  // Device function designed to be called by all threads of a given block
  template <bool debug = false>
  ALPAKA_FN_ACC static void hcalFastCluster_multiSeedParallel(
      const Acc1D& acc,
      ::reco::PFClusterParamsSoA::ConstView pfClusParams,
      const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
      int topoId,   // from selection
      int nSeeds,   // from selection
      int nRHTopo,  // from selection
      reco::PFRecHitDeviceCollection::ConstView pfRecHits,
      reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
      reco::PFClusterDeviceCollection::View clusterView,
      reco::PFRecHitFractionDeviceCollection::View fracView) {
    int tid = alpaka::getIdx<alpaka::Block,
                             alpaka::Threads>(  // Thread index corresponds to a single rechit of the topo cluster
        acc)[0u];

    int& nRHNotSeed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& topoSeedBegin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& stride = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& diff2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    auto& clusterPos = alpaka::declareSharedVar<Position4[100], __COUNTER__>(acc);
    auto& prevClusterPos = alpaka::declareSharedVar<Position4[100], __COUNTER__>(acc);
    auto& clusterEnergy = alpaka::declareSharedVar<float[100], __COUNTER__>(acc);
    auto& rhFracSum = alpaka::declareSharedVar<float[threadsPerBlockForClustering], __COUNTER__>(acc);
    auto& seeds = alpaka::declareSharedVar<int[100], __COUNTER__>(acc);
    auto& rechits = alpaka::declareSharedVar<int[threadsPerBlockForClustering], __COUNTER__>(acc);

    if (once_per_block(acc)) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = pfClusteringVars[topoId].topoSeedOffsets();
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1), 2.0);  // stopping tolerance * tolerance scaling
      stride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      iter = 0;
      notDone = true;

      int i = pfClusteringVars[topoSeedBegin].topoSeedList();

      if (topology.cutsFromDB()) {
        rhENormInv = (1.f / topology[pfRecHits[i].denseId()].noiseThreshold());
      } else {
        if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
          rhENormInv = pfClusParams.recHitEnergyNormInvHB_vec()[pfRecHits[i].depth() - 1];
        else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
          rhENormInv = pfClusParams.recHitEnergyNormInvHE_vec()[pfRecHits[i].depth() - 1];
        else {
          rhENormInv = 0.;
          printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
        }
      }
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    if (tid < nSeeds)
      seeds[tid] = pfClusteringVars[topoSeedBegin + tid].topoSeedList();
    if (tid < nRHNotSeed - 1)
      rechits[tid] =
          fracView[pfClusteringVars[pfClusteringVars[topoSeedBegin].topoSeedList()].seedFracOffsets() + tid + 1]
              .pfrhIdx();

    alpaka::syncBlockThreads(acc);  // all threads call sync

    if constexpr (debug) {
      if (once_per_block(acc)) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
        for (int s = 0; s < nSeeds; s++) {
          if (s != 0)
            printf(", ");
          printf("%d", getSeedRhIdx(seeds, s));
        }
        if (nRHTopo == nSeeds) {
          printf(")\n\n");
        } else {
          printf(") and other rechits (");
          for (int r = 1; r < nRHNotSeed; r++) {
            if (r != 1)
              printf(", ");
            if (r <= 0) {
              printf("Invalid rhNum (%d) for get RhFracIdx!\n", r);
            }
            printf("%d", rechits[r - 1]);
          }
          printf(")\n\n");
        }
      }
      alpaka::syncBlockThreads(acc);  // all (or none) threads call sync
    }

    // Set initial cluster position (energy) to seed rechit position (energy)
    if (tid < nSeeds) {
      int i = getSeedRhIdx(seeds, tid);
      clusterPos[tid] = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.0};
      prevClusterPos[tid] = clusterPos[tid];
      clusterEnergy[tid] = pfRecHits[i].energy();
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        fracView[pfClusteringVars[i].seedFracOffsets() + r + 1].pfrhIdx() = rechits[r];
        fracView[pfClusteringVars[i].seedFracOffsets() + r + 1].frac() = -1.;
      }
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    int rhThreadIdx = -1;
    Position4 rhThreadPos;
    if (tid < (nRHNotSeed - 1)) {
      rhThreadIdx = rechits[tid];  // Index when thread represents rechit
      rhThreadPos = Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
    }

    // Neighbors when threadIdx represents seed
    int seedThreadIdx = -1;
    Neighbours4 seedNeighbors = Neighbours4{-9, -9, -9, -9};
    float seedEnergy = -1.;
    Position4 seedInitClusterPos = Position4{0., 0., 0., 0.};
    if (tid < nSeeds) {
      if constexpr (debug)
        printf("tid: %d\n", tid);
      seedThreadIdx = getSeedRhIdx(seeds, tid);
      seedNeighbors = Neighbours4{pfRecHits[seedThreadIdx].neighbours()(0),
                                  pfRecHits[seedThreadIdx].neighbours()(1),
                                  pfRecHits[seedThreadIdx].neighbours()(2),
                                  pfRecHits[seedThreadIdx].neighbours()(3)};
      seedEnergy = pfRecHits[seedThreadIdx].energy();

      // Compute initial cluster position shift for seed
      updateClusterPos(pfClusParams, seedInitClusterPos, 1., seedThreadIdx, pfRecHits, rhENormInv);
    }

    do {
      if constexpr (debug) {
        if (once_per_block(acc))
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      // Reset rhFracSum
      rhFracSum[tid] = 0.;
      if (once_per_block(acc))
        diff2 = -1;

      if (tid < (nRHNotSeed - 1)) {
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5f * d2);

          rhFracSum[tid] += fraction;
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (tid < (nRHNotSeed - 1)) {
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5f * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            if (fracpct > cutoffFraction || (d2 < cutoffDistance && fracpct > pfClusParams.minFracToKeep())) {
              fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = fracpct;
            } else {
              fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = -1;
            }
          } else {
            fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = -1;
          }
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if constexpr (debug) {
        if (once_per_block(acc))
          printf("Computing cluster position for topoId %d\n", topoId);
      }

      // Reset cluster position and energy
      if (tid < nSeeds) {
        clusterPos[tid] = seedInitClusterPos;
        clusterEnergy[tid] = seedEnergy;
        if constexpr (debug) {
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
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Recalculate position
      if (tid < nSeeds) {
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = getRhFrac(pfClusteringVars, topoSeedBegin, fracView, tid, r + 1);

          if (frac > -0.5) {
            clusterEnergy[tid] += frac * pfRecHits[j].energy();

            if (nSeeds == 1 || j == seedNeighbors.x || j == seedNeighbors.y || j == seedNeighbors.z ||
                j == seedNeighbors.w)
              updateClusterPos(pfClusParams, clusterPos[tid], frac, j, pfRecHits, rhENormInv);
          }
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Position normalization
      if (tid < nSeeds) {
        if (clusterPos[tid].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[tid].x /= clusterPos[tid].w;
          clusterPos[tid].y /= clusterPos[tid].w;
          clusterPos[tid].z /= clusterPos[tid].w;

          if constexpr (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   tid,
                   seedThreadIdx,
                   clusterEnergy[tid],
                   clusterPos[tid].x,
                   clusterPos[tid].y,
                   clusterPos[tid].z);
        } else {
          if constexpr (debug)
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
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (tid < nSeeds) {
        float delta2 = dR2(prevClusterPos[tid], clusterPos[tid]);
        if constexpr (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", tid, seeds[tid], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[tid] = clusterPos[tid];  // Save clusterPos
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (once_per_block(acc)) {
        float tol2 = tol * tol;
        iter++;
        notDone = (diff2 > tol2) && ((unsigned int)iter < pfClusParams.maxIterations());
        if constexpr (debug) {
          if (diff2 > tol2)
            printf("\tTopoId %d has diff2 = %f greater than squared tolerance %f (continuing)\n", topoId, diff2, tol2);
          else if constexpr (debug)
            printf("\tTopoId %d has diff2 = %f LESS than squared tolerance %f (terminating!)\n", topoId, diff2, tol2);
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync
    } while (notDone);  // shared variable condition ensures synchronization is well defined
    if (once_per_block(acc))
      // Fill PFCluster-level info
      if (tid < nSeeds) {
        int rhIdx = pfClusteringVars[tid + pfClusteringVars[topoId].topoSeedOffsets()].topoSeedList();
        int seedIdx = pfClusteringVars[rhIdx].rhIdxToSeedIdx();
        clusterView[seedIdx].energy() = clusterEnergy[tid];
        clusterView[seedIdx].x() = clusterPos[tid].x;
        clusterView[seedIdx].y() = clusterPos[tid].y;
        clusterView[seedIdx].z() = clusterPos[tid].z;
      }
  }

  // Process very large exotic clusters, from nSeeds > 400 and non-seeds > 1500
  // Uses global memory access
  // Device function designed to be called by all threads of a given block
  template <bool debug = false>
  ALPAKA_FN_ACC static void hcalFastCluster_exotic(const Acc1D& acc,
                                                   ::reco::PFClusterParamsSoA::ConstView pfClusParams,
                                                   const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
                                                   int topoId,
                                                   int nSeeds,
                                                   int nRHTopo,
                                                   reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                                   reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                                   reco::PFClusterDeviceCollection::View clusterView,
                                                   reco::PFRecHitFractionDeviceCollection::View fracView,
                                                   Position4* __restrict__ globalClusterPos,
                                                   Position4* __restrict__ globalPrevClusterPos,
                                                   float* __restrict__ globalClusterEnergy,
                                                   float* __restrict__ globalRhFracSum,
                                                   int* __restrict__ globalSeeds,
                                                   int* __restrict__ globalRechits) {
    int& nRHNotSeed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& blockIdx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& topoSeedBegin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& stride = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& diff2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);

    blockIdx = maxTopoInput * alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    Position4* clusterPos = globalClusterPos + blockIdx;
    Position4* prevClusterPos = globalPrevClusterPos + blockIdx;
    float* clusterEnergy = globalClusterEnergy + blockIdx;
    float* rhFracSum = globalRhFracSum + blockIdx;
    int* seeds = globalSeeds + blockIdx;
    int* rechits = globalRechits + blockIdx;

    if (once_per_block(acc)) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = pfClusteringVars[topoId].topoSeedOffsets();
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1), 2.0);  // stopping tolerance * tolerance scaling
      stride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      iter = 0;
      notDone = true;

      int i = pfClusteringVars[topoSeedBegin].topoSeedList();

      if (topology.cutsFromDB()) {
        rhENormInv = (1.f / topology[pfRecHits[i].denseId()].noiseThreshold());
      } else {
        if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
          rhENormInv = pfClusParams.recHitEnergyNormInvHB_vec()[pfRecHits[i].depth() - 1];
        else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
          rhENormInv = pfClusParams.recHitEnergyNormInvHE_vec()[pfRecHits[i].depth() - 1];
        else {
          rhENormInv = 0.;
          printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
        }
      }
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    for (int n = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; n < nRHTopo; n += stride) {
      if (n < nSeeds)
        seeds[n] = pfClusteringVars[topoSeedBegin + n].topoSeedList();
      if (n < nRHNotSeed - 1)
        rechits[n] =
            fracView[pfClusteringVars[pfClusteringVars[topoSeedBegin].topoSeedList()].seedFracOffsets() + n + 1]
                .pfrhIdx();
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    if constexpr (debug) {
      if (once_per_block(acc)) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
        for (int s = 0; s < nSeeds; s++) {
          if (s != 0)
            printf(", ");
          printf("%d", getSeedRhIdx(seeds, s));
        }
        if (nRHTopo == nSeeds) {
          printf(")\n\n");
        } else {
          printf(") and other rechits (");
          for (int r = 1; r < nRHNotSeed; r++) {
            if (r != 1)
              printf(", ");
            if (r <= 0) {
              printf("Invalid rhNum (%d) for get RhFracIdx!\n", r);
            }
            printf("%d", rechits[r - 1]);
          }
          printf(")\n\n");
        }
      }
      alpaka::syncBlockThreads(acc);  // all (or none) threads call sync
    }

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
      int i = seeds[s];
      clusterPos[s] = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.0};
      prevClusterPos[s] = clusterPos[s];
      clusterEnergy[s] = pfRecHits[i].energy();
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        fracView[pfClusteringVars[i].seedFracOffsets() + r + 1].pfrhIdx() = rechits[r];
        fracView[pfClusteringVars[i].seedFracOffsets() + r + 1].frac() = -1.;
      }
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    do {
      if constexpr (debug) {
        if (once_per_block(acc))
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      if (once_per_block(acc))
        diff2 = -1;
      // Reset rhFracSum
      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += stride) {
        rhFracSum[tid] = 0.;
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5f * d2);

          rhFracSum[tid] += fraction;
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += stride) {
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5f * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            if (fracpct > cutoffFraction || (d2 < cutoffDistance && fracpct > pfClusParams.minFracToKeep())) {
              fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = fracpct;
            } else {
              fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = -1;
            }
          } else {
            fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = -1;
          }
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if constexpr (debug) {
        if (once_per_block(acc))
          printf("Computing cluster position for topoId %d\n", topoId);
      }

      // Reset cluster position and energy
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        int seedRhIdx = getSeedRhIdx(seeds, s);
        float norm = logf(pfRecHits[seedRhIdx].energy() * rhENormInv);
        clusterPos[s] = Position4{
            pfRecHits[seedRhIdx].x() * norm, pfRecHits[seedRhIdx].y() * norm, pfRecHits[seedRhIdx].z() * norm, norm};
        clusterEnergy[s] = pfRecHits[seedRhIdx].energy();
        if constexpr (debug) {
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
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Recalculate position
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        int seedRhIdx = getSeedRhIdx(seeds, s);
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = getRhFrac(pfClusteringVars, topoSeedBegin, fracView, s, r + 1);

          if (frac > -0.5) {
            clusterEnergy[s] += frac * pfRecHits[j].energy();

            if (nSeeds == 1 || j == pfRecHits[seedRhIdx].neighbours()(0) || j == pfRecHits[seedRhIdx].neighbours()(1) ||
                j == pfRecHits[seedRhIdx].neighbours()(2) || j == pfRecHits[seedRhIdx].neighbours()(3))
              updateClusterPos(pfClusParams, clusterPos[s], frac, j, pfRecHits, rhENormInv);
          }
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Position normalization
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        if (clusterPos[s].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[s].x /= clusterPos[s].w;
          clusterPos[s].y /= clusterPos[s].w;
          clusterPos[s].z /= clusterPos[s].w;

          if constexpr (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   s,
                   seeds[s],
                   clusterEnergy[s],
                   clusterPos[s].x,
                   clusterPos[s].y,
                   clusterPos[s].z);
        } else {
          if constexpr (debug)
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
      alpaka::syncBlockThreads(acc);  // all threads call sync

      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
        if constexpr (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[s] = clusterPos[s];  // Save clusterPos
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (once_per_block(acc)) {
        float tol2 = tol * tol;
        iter++;
        notDone = (diff2 > tol2) && ((unsigned int)iter < pfClusParams.maxIterations());
        if constexpr (debug) {
          if (diff2 > tol2)
            printf("\tTopoId %d has diff2 = %f greater than squared tolerance %f (continuing)\n", topoId, diff2, tol2);
          else if constexpr (debug)
            printf("\tTopoId %d has diff2 = %f LESS than squared tolerance %f (terminating!)\n", topoId, diff2, tol2);
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync
    } while (notDone);  // shared variable ensures synchronization is well defined
    if (once_per_block(acc))
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        int rhIdx = pfClusteringVars[s + pfClusteringVars[topoId].topoSeedOffsets()].topoSeedList();
        int seedIdx = pfClusteringVars[rhIdx].rhIdxToSeedIdx();
        clusterView[seedIdx].energy() = pfRecHits[s].energy();
        clusterView[seedIdx].x() = pfRecHits[s].x();
        clusterView[seedIdx].y() = pfRecHits[s].y();
        clusterView[seedIdx].z() = pfRecHits[s].z();
      }
    alpaka::syncBlockThreads(acc);  // all threads call sync
  }

  // Process clusters with up to 400 seeds and 1500 non seeds using shared memory
  // Device function designed to be called by all threads of a given block
  template <bool debug = false>
  ALPAKA_FN_ACC static void hcalFastCluster_multiSeedIterative(
      const Acc1D& acc,
      ::reco::PFClusterParamsSoA::ConstView pfClusParams,
      const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
      int topoId,
      int nSeeds,
      int nRHTopo,
      reco::PFRecHitDeviceCollection::ConstView pfRecHits,
      reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
      reco::PFClusterDeviceCollection::View clusterView,
      reco::PFRecHitFractionDeviceCollection::View fracView) {
    int& nRHNotSeed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& topoSeedBegin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& stride = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& diff2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);

    auto& clusterPos = alpaka::declareSharedVar<Position4[400], __COUNTER__>(acc);
    auto& prevClusterPos = alpaka::declareSharedVar<Position4[400], __COUNTER__>(acc);
    auto& clusterEnergy = alpaka::declareSharedVar<float[400], __COUNTER__>(acc);
    auto& rhFracSum = alpaka::declareSharedVar<float[1500], __COUNTER__>(acc);
    auto& seeds = alpaka::declareSharedVar<int[400], __COUNTER__>(acc);
    auto& rechits = alpaka::declareSharedVar<int[1500], __COUNTER__>(acc);

    if (once_per_block(acc)) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = pfClusteringVars[topoId].topoSeedOffsets();
      tol = pfClusParams.stoppingTolerance() *  // stopping tolerance * tolerance scaling
            powf(fmaxf(1.0, nSeeds - 1), 2.0);
      stride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      iter = 0;
      notDone = true;

      int i = pfClusteringVars[topoSeedBegin].topoSeedList();

      if (topology.cutsFromDB()) {
        rhENormInv = (1.f / topology[pfRecHits[i].denseId()].noiseThreshold());
      } else {
        if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
          rhENormInv = pfClusParams.recHitEnergyNormInvHB_vec()[pfRecHits[i].depth() - 1];
        else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
          rhENormInv = pfClusParams.recHitEnergyNormInvHE_vec()[pfRecHits[i].depth() - 1];
        else {
          rhENormInv = 0.;
          printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
        }
      }
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    for (int n = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; n < nRHTopo; n += stride) {
      if (n < nSeeds)
        seeds[n] = pfClusteringVars[topoSeedBegin + n].topoSeedList();
      if (n < nRHNotSeed - 1)
        rechits[n] =
            fracView[pfClusteringVars[pfClusteringVars[topoSeedBegin].topoSeedList()].seedFracOffsets() + n + 1]
                .pfrhIdx();
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    if constexpr (debug) {
      if (once_per_block(acc)) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
        for (int s = 0; s < nSeeds; s++) {
          if (s != 0)
            printf(", ");
          printf("%d", getSeedRhIdx(seeds, s));
        }
        if (nRHTopo == nSeeds) {
          printf(")\n\n");
        } else {
          printf(") and other rechits (");
          for (int r = 1; r < nRHNotSeed; r++) {
            if (r != 1)
              printf(", ");
            if (r <= 0) {
              printf("Invalid rhNum (%d) for get RhFracIdx!\n", r);
            }
            printf("%d", rechits[r - 1]);
          }
          printf(")\n\n");
        }
      }
      alpaka::syncBlockThreads(acc);  // all (or none) threads call sync
    }

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
      int i = seeds[s];
      clusterPos[s] = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.0};
      prevClusterPos[s] = clusterPos[s];
      clusterEnergy[s] = pfRecHits[i].energy();
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        fracView[pfClusteringVars[i].seedFracOffsets() + r + 1].pfrhIdx() = rechits[r];
        fracView[pfClusteringVars[i].seedFracOffsets() + r + 1].frac() = -1.;
      }
    }
    alpaka::syncBlockThreads(acc);  // all threads call sync

    do {
      if constexpr (debug) {
        if (once_per_block(acc))
          printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      if (once_per_block(acc))
        diff2 = -1;
      // Reset rhFracSum
      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += stride) {
        rhFracSum[tid] = 0.;
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5f * d2);

          rhFracSum[tid] += fraction;
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += stride) {
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5f * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            if (fracpct > cutoffFraction || (d2 < cutoffDistance && fracpct > pfClusParams.minFracToKeep())) {
              fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = fracpct;
            } else {
              fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = -1;
            }
          } else {
            fracView[pfClusteringVars[i].seedFracOffsets() + tid + 1].frac() = -1;
          }
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if constexpr (debug) {
        if (once_per_block(acc))
          printf("Computing cluster position for topoId %d\n", topoId);
      }

      // Reset cluster position and energy
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        int seedRhIdx = getSeedRhIdx(seeds, s);
        float norm = logf(pfRecHits[seedRhIdx].energy() * rhENormInv);
        clusterPos[s] = Position4{
            pfRecHits[seedRhIdx].x() * norm, pfRecHits[seedRhIdx].y() * norm, pfRecHits[seedRhIdx].z() * norm, norm};
        clusterEnergy[s] = pfRecHits[seedRhIdx].energy();
        if constexpr (debug) {
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
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Recalculate position
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        int seedRhIdx = getSeedRhIdx(seeds, s);
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = getRhFrac(pfClusteringVars, topoSeedBegin, fracView, s, r + 1);

          if (frac > -0.5) {
            clusterEnergy[s] += frac * pfRecHits[j].energy();

            if (nSeeds == 1 || j == pfRecHits[seedRhIdx].neighbours()(0) || j == pfRecHits[seedRhIdx].neighbours()(1) ||
                j == pfRecHits[seedRhIdx].neighbours()(2) || j == pfRecHits[seedRhIdx].neighbours()(3))
              updateClusterPos(pfClusParams, clusterPos[s], frac, j, pfRecHits, rhENormInv);
          }
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Position normalization
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        if (clusterPos[s].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[s].x /= clusterPos[s].w;
          clusterPos[s].y /= clusterPos[s].w;
          clusterPos[s].z /= clusterPos[s].w;

          if constexpr (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   s,
                   seeds[s],
                   clusterEnergy[s],
                   clusterPos[s].x,
                   clusterPos[s].y,
                   clusterPos[s].z);
        } else {
          if constexpr (debug)
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
      alpaka::syncBlockThreads(acc);  // all threads call sync

      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
        if constexpr (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[s] = clusterPos[s];  // Save clusterPos
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (once_per_block(acc)) {
        float tol2 = tol * tol;
        iter++;
        notDone = (diff2 > tol2) && ((unsigned int)iter < pfClusParams.maxIterations());
        if constexpr (debug) {
          if (diff2 > tol2)
            printf("\tTopoId %d has diff2 = %f greater than tolerance %f (continuing)\n", topoId, diff2, tol2);
          else if constexpr (debug)
            printf("\tTopoId %d has diff2 = %f LESS than tolerance %f (terminating!)\n", topoId, diff2, tol2);
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync
    } while (notDone);  // shared variable ensures synchronization is well defined
    if (once_per_block(acc))
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += stride) {
        int rhIdx = pfClusteringVars[s + pfClusteringVars[topoId].topoSeedOffsets()].topoSeedList();
        int seedIdx = pfClusteringVars[rhIdx].rhIdxToSeedIdx();
        clusterView[seedIdx].energy() = pfRecHits[s].energy();
        clusterView[seedIdx].x() = pfRecHits[s].x();
        clusterView[seedIdx].y() = pfRecHits[s].y();
        clusterView[seedIdx].z() = pfRecHits[s].z();
      }
  }

  // Seeding using local energy maxima
  class SeedingTopoThresh {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  const ::reco::PFClusterParamsSoA::ConstView pfClusParams,
                                  const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  uint32_t* __restrict__ nSeeds) const {
      const int nRH = pfRecHits.size();

      if (once_per_grid(acc)) {
        clusterView.size() = nRH;
      }

      for (auto i : uniform_elements(acc, nRH)) {
        // Initialize arrays
        pfClusteringVars[i].pfrh_isSeed() = 0;
        pfClusteringVars[i].rhCount() = 0;
        pfClusteringVars[i].topoSeedCount() = 0;
        pfClusteringVars[i].topoRHCount() = 0;
        pfClusteringVars[i].seedFracOffsets() = -1;
        pfClusteringVars[i].topoSeedOffsets() = -1;
        pfClusteringVars[i].topoSeedList() = -1;
        clusterView[i].seedRHIdx() = -1;

        int layer = pfRecHits[i].layer();
        int depthOffset = pfRecHits[i].depth() - 1;
        float energy = pfRecHits[i].energy();
        Position3 pos = Position3{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z()};
        float seedThreshold = 9999.;
        float topoThreshold = 9999.;

        if (topology.cutsFromDB()) {
          seedThreshold = topology[pfRecHits[i].denseId()].seedThreshold();
          topoThreshold = topology[pfRecHits[i].denseId()].noiseThreshold();
        } else {
          if (layer == PFLayer::HCAL_BARREL1) {
            seedThreshold = pfClusParams.seedEThresholdHB_vec()[depthOffset];
            topoThreshold = pfClusParams.topoEThresholdHB_vec()[depthOffset];
          } else if (layer == PFLayer::HCAL_ENDCAP) {
            seedThreshold = pfClusParams.seedEThresholdHE_vec()[depthOffset];
            topoThreshold = pfClusParams.topoEThresholdHE_vec()[depthOffset];
          }
        }

        // cmssdt.cern.ch/lxr/source/DataFormats/ParticleFlowReco/interface/PFRecHit.h#0108
        float pt2 = energy * energy * (pos.x * pos.x + pos.y * pos.y) / (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

        // Seeding threshold test
        if ((layer == PFLayer::HCAL_BARREL1 && energy > seedThreshold && pt2 > pfClusParams.seedPt2ThresholdHB()) ||
            (layer == PFLayer::HCAL_ENDCAP && energy > seedThreshold && pt2 > pfClusParams.seedPt2ThresholdHE())) {
          pfClusteringVars[i].pfrh_isSeed() = 1;
          for (int k = 0; k < 4; k++) {  // Does this seed candidate have a higher energy than four neighbours
            if (pfRecHits[i].neighbours()(k) < 0)
              continue;
            if (energy < pfRecHits[pfRecHits[i].neighbours()(k)].energy()) {
              pfClusteringVars[i].pfrh_isSeed() = 0;
              break;
            }
          }
          if (pfClusteringVars[i].pfrh_isSeed())
            alpaka::atomicAdd(acc, nSeeds, 1u);
        }
        // Topo clustering threshold test

        if ((layer == PFLayer::HCAL_ENDCAP && energy > topoThreshold) ||
            (layer == PFLayer::HCAL_BARREL1 && energy > topoThreshold)) {
          pfClusteringVars[i].pfrh_passTopoThresh() = true;
          pfClusteringVars[i].pfrh_topoId() = i;
        } else {
          pfClusteringVars[i].pfrh_passTopoThresh() = false;
          pfClusteringVars[i].pfrh_topoId() = -1;
        }
      }
    }
  };

  // Preparation of topo inputs. Initializing topoId, egdeIdx, nEdges, edgeList
  class PrepareTopoInputs {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars,
                                  uint32_t* __restrict__ nSeeds) const {
      const int nRH = pfRecHits.size();

      if (once_per_grid(acc)) {
        pfClusteringVars.nEdges() = nRH * 8;
        pfClusteringEdgeVars[nRH].pfrh_edgeIdx() = nRH * 8;
      }
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nRH)) {
        pfClusteringEdgeVars[i].pfrh_edgeIdx() = i * 8;
        pfClusteringVars[i].pfrh_topoId() = 0;
        for (int j = 0; j < 8; j++) {  // checking if neighbours exist and assigning neighbours as edges
          if (pfRecHits[i].neighbours()(j) == -1)
            pfClusteringEdgeVars[i * 8 + j].pfrh_edgeList() = i;
          else
            pfClusteringEdgeVars[i * 8 + j].pfrh_edgeList() = pfRecHits[i].neighbours()(j);
        }
      }

      return;
    }
  };

  // Contraction in a single block
  class TopoClusterContraction {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  uint32_t* __restrict__ nSeeds,
                                  uint32_t* __restrict__ nRHF) const {
      const int nRH = pfRecHits.size();
      int& totalSeedOffset = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& totalSeedFracOffset = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      // rhCount, topoRHCount, topoSeedCount initialized earlier
      if (once_per_block(acc)) {
        pfClusteringVars.nTopos() = 0;
        pfClusteringVars.nRHFracs() = 0;
        totalSeedOffset = 0;
        totalSeedFracOffset = 0;
        pfClusteringVars.pcrhFracSize() = 0;
      }

      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Now determine the number of seeds and rechits in each topo cluster [topoRHCount, topoSeedCount]
      // Also get the list of topoIds (smallest rhIdx of each topo cluser)
      for (int rhIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; rhIdx < nRH;
           rhIdx += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
        pfClusteringVars[rhIdx].rhIdxToSeedIdx() = -1;
        int topoId = pfClusteringVars[rhIdx].pfrh_topoId();
        if (topoId > -1) {
          // Valid topo cluster
          alpaka::atomicAdd(acc, &pfClusteringVars[topoId].topoRHCount(), 1);
          // Valid topoId not counted yet
          if (topoId == rhIdx) {  // For every topo cluster, there is one rechit that meets this condition.
            int topoIdx = alpaka::atomicAdd(acc, &pfClusteringVars.nTopos(), 1);
            pfClusteringVars[topoIdx].topoIds() =
                topoId;  // topoId: the smallest index of rechits that belong to a topo cluster.
          }
          // This is a cluster seed
          if (pfClusteringVars[rhIdx].pfrh_isSeed()) {  // # of seeds in this topo cluster
            alpaka::atomicAdd(acc, &pfClusteringVars[topoId].topoSeedCount(), 1);
          }
        }
      }

      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Determine offsets for topo ID seed array [topoSeedOffsets]
      for (int topoId = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; topoId < nRH;
           topoId += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
        if (pfClusteringVars[topoId].topoSeedCount() > 0) {
          // This is a valid topo ID
          int offset = alpaka::atomicAdd(acc, &totalSeedOffset, pfClusteringVars[topoId].topoSeedCount());
          pfClusteringVars[topoId].topoSeedOffsets() = offset;
        }
      }
      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Fill arrays of rechit indicies for each seed [topoSeedList] and rhIdx->seedIdx conversion for each seed [rhIdxToSeedIdx]
      // Also fill seedRHIdx, topoId, depth
      for (int rhIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; rhIdx < nRH;
           rhIdx += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
        int topoId = pfClusteringVars[rhIdx].pfrh_topoId();
        if (pfClusteringVars[rhIdx].pfrh_isSeed()) {
          // Valid topo cluster and this rhIdx corresponds to a seed
          int k = alpaka::atomicAdd(acc, &pfClusteringVars[topoId].rhCount(), 1);
          int seedIdx = pfClusteringVars[topoId].topoSeedOffsets() + k;
          if ((unsigned int)seedIdx >= *nSeeds)
            printf("Warning(contraction) %8d > %8d should not happen, check topoId: %d has %d rh\n",
                   seedIdx,
                   *nSeeds,
                   topoId,
                   k);
          pfClusteringVars[seedIdx].topoSeedList() = rhIdx;
          pfClusteringVars[rhIdx].rhIdxToSeedIdx() = seedIdx;
          clusterView[seedIdx].topoId() = topoId;
          clusterView[seedIdx].seedRHIdx() = rhIdx;
          clusterView[seedIdx].depth() = pfRecHits[rhIdx].depth();
        }
      }

      alpaka::syncBlockThreads(acc);  // all threads call sync

      // Determine seed offsets for rechit fraction array
      for (int rhIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; rhIdx < nRH;
           rhIdx += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
        pfClusteringVars[rhIdx].rhCount() = 1;  // Reset this counter array

        int topoId = pfClusteringVars[rhIdx].pfrh_topoId();
        if (pfClusteringVars[rhIdx].pfrh_isSeed() && topoId > -1) {
          // Allot the total number of rechits for this topo cluster for rh fractions
          int offset = alpaka::atomicAdd(acc, &totalSeedFracOffset, pfClusteringVars[topoId].topoRHCount());

          // Add offset for this PF cluster seed
          pfClusteringVars[rhIdx].seedFracOffsets() = offset;

          // Store recHitFraction offset & size information for each seed
          clusterView[pfClusteringVars[rhIdx].rhIdxToSeedIdx()].rhfracOffset() =
              pfClusteringVars[rhIdx].seedFracOffsets();
          clusterView[pfClusteringVars[rhIdx].rhIdxToSeedIdx()].rhfracSize() =
              pfClusteringVars[topoId].topoRHCount() - pfClusteringVars[topoId].topoSeedCount() + 1;
        }
      }

      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (once_per_block(acc)) {
        pfClusteringVars.pcrhFracSize() = totalSeedFracOffset;
        pfClusteringVars.nRHFracs() = totalSeedFracOffset;
        clusterView.nRHFracs() = totalSeedFracOffset;
        *nRHF = totalSeedFracOffset;
        clusterView.nSeeds() = *nSeeds;
        clusterView.nTopos() = pfClusteringVars.nTopos();

        if (pfClusteringVars.pcrhFracSize() > 200000)  // Warning in case the fraction is too large
          printf("At the end of topoClusterContraction, found large *pcrhFracSize = %d\n",
                 pfClusteringVars.pcrhFracSize());
      }
    }
  };

  // Prefill the rechit index for all PFCluster fractions
  // Optimized for GPU parallel, but works on any backend
  class FillRhfIndex {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFRecHitFractionDeviceCollection::View fracView) const {
      const int nRH = pfRecHits.size();

      for (auto index : uniform_elements_nd(acc, {nRH, nRH})) {
        const int i = index[0u];  // i is a seed index
        const int j = index[1u];  // j is NOT a seed
        int topoId = pfClusteringVars[i].pfrh_topoId();
        if (topoId > -1 && pfClusteringVars[i].pfrh_isSeed() && topoId == pfClusteringVars[j].pfrh_topoId()) {
          if (!pfClusteringVars[j].pfrh_isSeed()) {  // NOT a seed
            int k = alpaka::atomicAdd(
                acc, &pfClusteringVars[i].rhCount(), 1);  // Increment the number of rechit fractions for this seed
            auto fraction = fracView[pfClusteringVars[i].seedFracOffsets() + k];
            fraction.pfrhIdx() = j;
            fraction.pfcIdx() = pfClusteringVars[i].rhIdxToSeedIdx();
          } else if (i == j) {  // i==j is a seed rechit index
            auto seed = fracView[pfClusteringVars[i].seedFracOffsets()];
            seed.pfrhIdx() = j;
            seed.frac() = 1;
            seed.pfcIdx() = pfClusteringVars[i].rhIdxToSeedIdx();
          }
        }
      }
    }
  };

  class FastCluster {
  public:
    template <bool debug = false, typename = std::enable_if<!std::is_same_v<Device, alpaka::DevCpu>>>
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  const ::reco::PFClusterParamsSoA::ConstView pfClusParams,
                                  const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  reco::PFRecHitFractionDeviceCollection::View fracView) const {
      const int nRH = pfRecHits.size();
      int& topoId = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nRHTopo = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nSeeds = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      if (once_per_block(acc)) {
        topoId = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        nRHTopo = pfClusteringVars[topoId].topoRHCount();
        nSeeds = pfClusteringVars[topoId].topoSeedCount();
      }

      alpaka::syncBlockThreads(acc);  // all threads call sync

      if (topoId < nRH && nRHTopo > 0 && nSeeds > 0) {
        if (nRHTopo == nSeeds) {
          // PF cluster is isolated seed. No iterations needed
          if (once_per_block(acc)) {
            // Fill PFCluster-level information
            int rhIdx = pfClusteringVars[pfClusteringVars[topoId].topoSeedOffsets()].topoSeedList();
            int seedIdx = pfClusteringVars[rhIdx].rhIdxToSeedIdx();
            clusterView[seedIdx].energy() = pfRecHits[rhIdx].energy();
            clusterView[seedIdx].x() = pfRecHits[rhIdx].x();
            clusterView[seedIdx].y() = pfRecHits[rhIdx].y();
            clusterView[seedIdx].z() = pfRecHits[rhIdx].z();
          }
          // singleSeed and multiSeedParallel functions work only for GPU backend
        } else if ((not std::is_same_v<Device, alpaka::DevCpu>) && nSeeds == 1) {
          // Single seed cluster
          hcalFastCluster_singleSeed(
              acc, pfClusParams, topology, topoId, nRHTopo, pfRecHits, pfClusteringVars, clusterView, fracView);
        } else if ((not std::is_same_v<Device, alpaka::DevCpu>) && nSeeds <= 100 &&
                   nRHTopo - nSeeds < threadsPerBlockForClustering) {
          hcalFastCluster_multiSeedParallel(
              acc, pfClusParams, topology, topoId, nSeeds, nRHTopo, pfRecHits, pfClusteringVars, clusterView, fracView);
        } else if (nSeeds <= 400 && nRHTopo - nSeeds <= 1500) {
          // nSeeds value must match exotic in FastClusterExotic
          hcalFastCluster_multiSeedIterative(
              acc, pfClusParams, topology, topoId, nSeeds, nRHTopo, pfRecHits, pfClusteringVars, clusterView, fracView);
        } else {
          if constexpr (debug) {
            if (once_per_block(acc))
              printf("Topo cluster %d has %d seeds and %d rechits. Will be processed in next kernel.\n",
                     topoId,
                     nSeeds,
                     nRHTopo);
          }
        }
      }
    }
  };

  // Process very large, exotic topo clusters
  class FastClusterExotic {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  const ::reco::PFClusterParamsSoA::ConstView pfClusParams,
                                  const reco::PFRecHitHCALTopologyDeviceCollection::ConstView topology,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  reco::PFRecHitFractionDeviceCollection::View fracView,
                                  Position4* __restrict__ globalClusterPos,
                                  Position4* __restrict__ globalPrevClusterPos,
                                  float* __restrict__ globalClusterEnergy,
                                  float* __restrict__ globalRhFracSum,
                                  int* __restrict__ globalSeeds,
                                  int* __restrict__ globalRechits) const {
      const int nRH = pfRecHits.size();
      for (int topoId = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]; topoId < nRH;
           topoId += blocksForExoticClusters) {
        int nRHTopo = pfClusteringVars[topoId].topoRHCount();
        int nSeeds = pfClusteringVars[topoId].topoSeedCount();

        // nSeeds value must match multiSeedIterative in FastCluster
        if (nRHTopo > 0 && (nSeeds > 400 || nRHTopo - nSeeds > 1500)) {
          hcalFastCluster_exotic(acc,
                                 pfClusParams,
                                 topology,
                                 topoId,
                                 nSeeds,
                                 nRHTopo,
                                 pfRecHits,
                                 pfClusteringVars,
                                 clusterView,
                                 fracView,
                                 globalClusterPos,
                                 globalPrevClusterPos,
                                 globalClusterEnergy,
                                 globalRhFracSum,
                                 globalSeeds,
                                 globalRechits);
        }
        alpaka::syncBlockThreads(acc);  // all threads call sync
      }
    }
  };

  PFClusterProducerKernel::PFClusterProducerKernel(Queue& queue)
      : nSeeds(cms::alpakatools::make_device_buffer<uint32_t>(queue)),
        globalClusterPos(
            cms::alpakatools::make_device_buffer<Position4[]>(queue, blocksForExoticClusters * maxTopoInput)),
        globalPrevClusterPos(
            cms::alpakatools::make_device_buffer<Position4[]>(queue, blocksForExoticClusters * maxTopoInput)),
        globalClusterEnergy(
            cms::alpakatools::make_device_buffer<float[]>(queue, blocksForExoticClusters * maxTopoInput)),
        globalRhFracSum(cms::alpakatools::make_device_buffer<float[]>(queue, blocksForExoticClusters * maxTopoInput)),
        globalSeeds(cms::alpakatools::make_device_buffer<int[]>(queue, blocksForExoticClusters * maxTopoInput)),
        globalRechits(cms::alpakatools::make_device_buffer<int[]>(queue, blocksForExoticClusters * maxTopoInput)) {
    alpaka::memset(queue, nSeeds, 0x00);  // Reset nSeeds
  }

  void PFClusterProducerKernel::seedTopoAndContract(Queue& queue,
                                                    const ::reco::PFClusterParamsSoA::ConstView params,
                                                    const reco::PFRecHitHCALTopologyDeviceCollection& topology,
                                                    reco::PFClusteringVarsDeviceCollection& pfClusteringVars,
                                                    reco::PFClusteringEdgeVarsDeviceCollection& pfClusteringEdgeVars,
                                                    const reco::PFRecHitDeviceCollection& pfRecHits,
                                                    int nRH,
                                                    reco::PFClusterDeviceCollection& pfClusters,
                                                    uint32_t* __restrict__ nRHF) {
    const int threadsPerBlock = 256;
    const int blocks = divide_up_by(nRH, threadsPerBlock);

    // seedingTopoThresh
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        SeedingTopoThresh{},
                        pfClusteringVars.view(),
                        params,
                        topology.view(),
                        pfRecHits.view(),
                        pfClusters.view(),
                        nSeeds.data());
    // prepareTopoInputs
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        PrepareTopoInputs{},
                        pfRecHits.view(),
                        pfClusteringVars.view(),
                        pfClusteringEdgeVars.view(),
                        nSeeds.data());
    // ECLCC
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCInit{},
                        pfRecHits.view(),
                        pfClusteringVars.view(),
                        pfClusteringEdgeVars.view());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCCompute1{},
                        pfRecHits.view(),
                        pfClusteringVars.view(),
                        pfClusteringEdgeVars.view());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCFlatten{},
                        pfRecHits.view(),
                        pfClusteringVars.view(),
                        pfClusteringEdgeVars.view());
    // topoClusterContraction
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(1, threadsPerBlockForClustering),
                        TopoClusterContraction{},
                        pfRecHits.view(),
                        pfClusteringVars.view(),
                        pfClusters.view(),
                        nSeeds.data(),
                        nRHF);
  }

  void PFClusterProducerKernel::cluster(Queue& queue,
                                        const ::reco::PFClusterParamsSoA::ConstView params,
                                        const reco::PFRecHitHCALTopologyDeviceCollection& topology,
                                        reco::PFClusteringVarsDeviceCollection& pfClusteringVars,
                                        reco::PFClusteringEdgeVarsDeviceCollection& pfClusteringEdgeVars,
                                        const reco::PFRecHitDeviceCollection& pfRecHits,
                                        int nRH,
                                        reco::PFClusterDeviceCollection& pfClusters,
                                        reco::PFRecHitFractionDeviceCollection& pfrhFractions) {
    // fillRhfIndex
    alpaka::exec<Acc2D>(queue,
                        make_workdiv<Acc2D>({divide_up_by(nRH, 32), divide_up_by(nRH, 32)}, {32, 32}),
                        FillRhfIndex{},
                        pfRecHits.view(),
                        pfClusteringVars.view(),
                        pfrhFractions.view());

    // Run fastCluster
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(nRH, threadsPerBlockForClustering),
                        FastCluster{},
                        pfRecHits.view(),
                        params,
                        topology.view(),
                        pfClusteringVars.view(),
                        pfClusters.view(),
                        pfrhFractions.view());
    // exotic clustering kernel
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocksForExoticClusters,
                                            threadsPerBlockForClustering),  // uses 4 blocks to minimize memory usage
                        FastClusterExotic{},
                        pfRecHits.view(),
                        params,
                        topology.view(),
                        pfClusteringVars.view(),
                        pfClusters.view(),
                        pfrhFractions.view(),
                        globalClusterPos.data(),
                        globalPrevClusterPos.data(),
                        globalClusterEnergy.data(),
                        globalRhFracSum.data(),
                        globalSeeds.data(),
                        globalRechits.data());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

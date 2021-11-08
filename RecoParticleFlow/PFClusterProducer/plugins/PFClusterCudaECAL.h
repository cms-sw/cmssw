#ifndef PFClusterCudaECAL_h
#define PFClusterCudaECAL_h
#include "RecoParticleFlow/PFClusterProducer/plugins/CudaPFCommon.h"
#include <Eigen/Dense>
#include <cuda.h>

namespace PFClusterCudaECAL {
  void initializeCudaConstants(const PFClustering::common::CudaECALConstants& cudaConstants,
                               const cudaStream_t cudaStream = 0);
  
  void initializeCudaConstants(const float h_showerSigma2,
                               const float h_recHitEnergyNormInvEB,
                               const float h_recHitEnergyNormInvEE, 
                               const float h_minFracToKeep,
                               const float h_minFracTot,
                               const uint32_t   h_maxIterations,
                               const float h_stoppingTolerance,
                               const bool  h_excludeOtherSeeds, 
                               const float h_seedEThresholdEB,
                               const float h_seedEThresholdEE,
                               const float h_seedPt2ThresholdEB,
                               const float h_seedPt2hresholdEE, 
                               const float h_topoEThresholdEB, 
                               const float h_topoEThresholdEE,
                               const int   h_nNeigh,
                               const PFClustering::common::PosCalcConfig h_posCalcConfig,
                               const PFClustering::common::ECALPosDepthCalcConfig h_posConvCalcConfig,
                               cudaStream_t cudaStream = 0
                               );


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
                int* pcrhFracSize
                );

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
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]
                );

  
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
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float* timer = nullptr
                );

}  // namespace cudavectors

#endif  // ClusterCudaECAL_h

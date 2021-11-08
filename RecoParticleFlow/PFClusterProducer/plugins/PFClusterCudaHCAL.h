#ifndef PFClusterCudaHCAL_h
#define PFClusterCudaHCAL_h
#include <Eigen/Dense>
#include "RecoParticleFlow/PFClusterProducer/plugins/CudaPFCommon.h"

namespace PFClusterCudaHCAL { 
  
  void initializeCudaConstants(const PFClustering::common::CudaHCALConstants& cudaConstants, 
                               const cudaStream_t cudaStream = 0);

void PFRechitToPFCluster_HCALV2(size_t size, 
				const float* __restrict__ pfrh_x,
				const float* __restrict__ pfrh_y,
				const float* __restrict__ pfrh_z,
				const float* __restrict__ pfrh_energy,
				const float* __restrict__ pfrh_pt2,
				int* pfrh_isSeed, 
				bool* pfrh_passTopoThresh, 
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
				const int* __restrict__ neigh8_Ind, 				
				const int* __restrict__ neigh4_Ind, 				
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]  
                );

void PFRechitToPFCluster_HCALV2(size_t size, 
				const float* __restrict__ pfrh_x,
				const float* __restrict__ pfrh_y,
				const float* __restrict__ pfrh_z,
				const float* __restrict__ pfrh_energy,
				const float* __restrict__ pfrh_pt2,
				int* pfrh_isSeed, 
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
				const int* __restrict__ neigh8_Ind, 				
				const int* __restrict__ neigh4_Ind, 				
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]  
                );



void PFRechitToPFCluster_HCAL_CCLClustering(cudaStream_t cudaStream,
                int nRH,
                int nEdges,
				const float* __restrict__ pfrh_x,
				const float* __restrict__ pfrh_y,
				const float* __restrict__ pfrh_z,
				const float* __restrict__ pfrh_energy,
				const float* __restrict__ pfrh_pt2,
			    int* pfrh_isSeed, 
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
				const int* __restrict__ neigh8_Ind, 				
				const int* __restrict__ neigh4_Ind, 				
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
                float4* pfc_pos,
                float4* pfc_prevPos,
                float* pfc_energy,
                float (&timer)[8],
                int* topoIter,
                int* pfcIter,
                int* pcrhFracSize
                );


void PFRechitToPFCluster_HCAL_serialize(size_t size, 
				const float* __restrict__ pfrh_x,
				const float* __restrict__ pfrh_y,
				const float* __restrict__ pfrh_z,
				const float* __restrict__ pfrh_energy,
				const float* __restrict__ pfrh_pt2,
				int* pfrh_isSeed, 
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
				const int* __restrict__ neigh8_Ind, 				
				const int* __restrict__ neigh4_Ind, 				
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float* timer = nullptr 
				);

}  // namespace PFClusterCudaHCAL 

#endif  // ClusterCudaHCAL_h

# SiStripClusterSoA
The `SiStripClusterHost`/`SiStripClusterDevice` is a portable collection based on the `SiStripClusterSoALayout` (`DataFormats/SiStripClusterSoA/interface/SiStripClusterSoA.h`).

It is used to store the collection of SiStrip cluster candidates from the heterogeneous SiStripClusterizer module (`RecoLocalTracker/SiStripClusterizer`).

## Data members
The fields in the structure have the following meaning:

| SoA type | C-type | Name | Description |
| --- | --- | --- | --- |
| column | uint32_t | clusterIndex | Index for the first strip amplitude in this cluster candidate, to be fetched by the Digi collection |
| column | uint16_t | clusterSize | Cluster cand. strip count |
| column | uint32_t | clusterDetId | Cluster cand. detector ID |
| column | uint16_t | firstStrip | First strip ID of the cluster cand. |
| column | bool | candidateAccepted | Is the cluster candidate a good one? [^ThreeThresholdAlgorithm.cc#L103-L107] |
| column | float | barycenter | Cluster cand. barycenter [^SiStripCluster.h#L164] |
| column | float | charge | Cluster cand. charge [^SiStripCluster.h#L152-L159] |
| column | uint32_t | candidateAcceptedPrefix | Prefix sum of the candidateAccepted colum, used to index the good clusters |
| scalar | uint32_t | nClusterCandidates | Number of cluster candidates in the collection [^noteCand] |
| scalar | uint32_t | maxClusterSize | Max number of contiguous strips for clustering (setup) |

<!--  -->
[^ThreeThresholdAlgorithm.cc#L103-L107]: [RecoLocalTracker/SiStripClusterizer/src/ThreeThresholdAlgorithm.cc](https://github.com/cms-sw/cmssw/blob/CMSSW_16_0_X/RecoLocalTracker/SiStripClusterizer/src/ThreeThresholdAlgorithm.cc#L103-L107)
[^SiStripCluster.h#L164]: [DataFormats/SiStripCluster/interface/SiStripCluster.h#L164](https://github.com/cms-sw/cmssw/blob/CMSSW_16_0_X/DataFormats/SiStripCluster/interface/SiStripCluster.h#L164)
[^SiStripCluster.h#L152-L159]: [DataFormats/SiStripCluster/interface/SiStripCluster.h#L152-L159](https://github.com/cms-sw/cmssw/blob/CMSSW_16_0_X/DataFormats/SiStripCluster/interface/SiStripCluster.h#L152-L159)
[^noteCand]: this is typically lower than the collection pre-allocated size (i.e., from `collection->metadata().size()`). It indicates the number of non-contiguous strip seeds over which the clustering is perfomed. It can be used while loop over the collection to break earlier than the collection size.

# SiStripClusterSoA
The `SiStripClusterHost`/`SiStripClusterDevice` is a portable collection based on the `SiStripClusterSoALayout` (`DataFormats/SiStripClusterSoA/interface/SiStripClusterSoA.h`).

It is used to store the collection of SiStrip cluster candidates from the heterogeneous SiStripClusterizer module (`RecoLocalTracker/SiStripClusterizer`).

## Data members
The fields in the structure have the following meaning:

| SoA type | C-type | Name | Description |
| --- | --- | --- | --- |
| column | uint32_t | clusterIndex | Index for the first strip amplitude in this cluster candidate, to be fetched by the Digi collection |
| column | uint16_t | clusterSize | Numbers of strips in the cluster candidate |
| column | uint32_t | clusterDetId | Cluster candidate corresponding detID |
| column | uint16_t | firstStrip | Value of the first strip ID |
| column | bool | candidateAccepted | Is this a good candidate? |
| column | float | barycenter | Barycenter, as in [1] |
| column | float | charge | Charge, as in [1] |
| column | uint32_t | candidateAcceptedPrefix | Prefix sum of the candidateAccepted colum, used to index the good clusters |
| scalar | uint32_t | nClusterCandidates | Number of cluster candidates in the collection |
| scalar | uint32_t | maxClusterSize | Max number of contiguous strips for clustering (setup) |

## References
[1] DataFormats/SiStripCluster/interface/SiStripCluster.h

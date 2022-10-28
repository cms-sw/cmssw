import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *

siStripClusters = cms.EDProducer("SiStripClusterizer",
                               Clusterizer = DefaultClusterizer,
                               DigiProducersList = cms.VInputTag(
    cms.InputTag('siStripDigis','ZeroSuppressed'),
    cms.InputTag('siStripZeroSuppression','VirginRaw'),
    cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
    cms.InputTag('siStripZeroSuppression','ScopeMode')),
                               )

from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters
from RecoLocalTracker.SiStripClusterizer.SiStripApprox2Clusters_cfi import SiStripApprox2Clusters
SiStripApprox2Clusters.inputApproxClusters = 'SiStripClusters2ApproxClusters'
approxSiStripClusters.toModify(SiStripApprox2Clusters, inputApproxClusters = 'SiStripClusters2ApproxClustersHLT')
approxSiStripClusters.toReplaceWith(siStripClusters,SiStripApprox2Clusters)

# The SiStripClusters are not used anymore in phase2 tracking
# This part has to be clean up when they will be officially removed from the entire flow
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siStripClusters, # FIXME
  DigiProducersList = [ 'simSiStripDigis:ZeroSuppressed',
                        'siStripZeroSuppression:VirginRaw',
                        'siStripZeroSuppression:ProcessedRaw',
                        'siStripZeroSuppression:ScopeMode']
)

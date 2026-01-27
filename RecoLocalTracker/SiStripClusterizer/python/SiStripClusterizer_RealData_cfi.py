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
approxSiStripClusters.toModify(SiStripApprox2Clusters, inputApproxClusters = 'hltSiStripClusters2ApproxClusters')
approxSiStripClusters.toReplaceWith(siStripClusters,SiStripApprox2Clusters)

##
## Modify for the tau embedding methods cleaning step
##
from Configuration.ProcessModifiers.tau_embedding_cleaning_cff import tau_embedding_cleaning
from TauAnalysis.MCEmbeddingTools.Cleaning_RECO_cff import tau_embedding_siStripClusters_cleaner
tau_embedding_cleaning.toReplaceWith(siStripClusters, tau_embedding_siStripClusters_cleaner)
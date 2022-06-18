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


from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(siStripClusters,
                       cms.EDProducer("SiStripApprox2Clusters",
                            inputApproxClusters = cms.InputTag('SiStripClusters2ApproxClusters')
                       ) 
)

# The SiStripClusters are not used anymore in phase2 tracking
# This part has to be clean up when they will be officially removed from the entire flow
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siStripClusters, # FIXME
  DigiProducersList = [ 'simSiStripDigis:ZeroSuppressed',
                        'siStripZeroSuppression:VirginRaw',
                        'siStripZeroSuppression:ProcessedRaw',
                        'siStripZeroSuppression:ScopeMode']
)

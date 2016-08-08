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

from Configuration.StandardSequences.Eras import eras
# The SiStripClusters are not used anymore in phase2 tracking
# This part has to be clean up when they will be officially removed from the entire flow
eras.phase2_tracker.toModify(siStripClusters, # FIXME
  DigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','ZeroSuppressed'),
                                     cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                     cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                     cms.InputTag('siStripZeroSuppression','ScopeMode'))
)


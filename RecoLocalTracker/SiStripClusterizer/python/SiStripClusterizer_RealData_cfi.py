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
# Need these until phase2 pixel templates are used
eras.phase2_tracker.toModify(siStripClusters, # FIXME
  DigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','ZeroSuppressed'),
                                     cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                     cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                     cms.InputTag('siStripZeroSuppression','ScopeMode'))
)


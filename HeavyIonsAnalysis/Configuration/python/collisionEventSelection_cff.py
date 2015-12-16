import FWCore.ParameterSet.Config as cms

# Coincidence of HF towers above threshold
from HeavyIonsAnalysis.Configuration.hfCoincFilter_cff import *

# Selection of at least a two-track fitted vertex
primaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2 && tracksSize >= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
)

# Cluster-shape filter re-run offline
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from HLTrigger.special.hltPixelClusterShapeFilter_cfi import *
hltPixelClusterShapeFilter.inputTag = "siPixelRecHits"

# Cluster-shape filter re-run offline from ClusterCompatibility object
from HeavyIonsAnalysis.EventAnalysis.HIClusterCompatibilityFilter_cfi import *


# Reject BSC beam halo L1 technical bits
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
noBSChalo = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39)')
)

collisionEventSelection = cms.Sequence(hfCoincFilter3 *
                                       primaryVertexFilter *
                                       siPixelRecHits *
                                       hltPixelClusterShapeFilter)

collisionEventSelectionAOD = cms.Sequence(hfCoincFilter3 *
                                          primaryVertexFilter *
                                          clusterCompatibilityFilter)

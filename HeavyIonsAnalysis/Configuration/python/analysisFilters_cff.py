import FWCore.ParameterSet.Config as cms

######################################################
# A set of filters for heavy-ion analysis skimming:
#   (1) L1 algo and technical bits
#   (2) HLT paths
#   (3) Event selection
#   (4) Physics objects


##### L1 selections #####
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed

# tech bit 0 - BPTX coincidence
bptxAnd = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('0')
    )

# tech bit 34 - BSC single-side - bits 36,37,38,39 - BSC beam halo veto
bscOr = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('(34) AND NOT (36 OR 37 OR 38 OR 39)')
    )

# algo bit 124 - BSC OR + BPTX OR
bscOrBptxOr = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(False),
    L1SeedsLogicalExpression = cms.string('L1_BscMinBiasOR_BptxPlusORMinus')
    )


##### HLT selections #####
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel

# jet trigger
hltJets = hltHighLevel.clone(
    HLTPaths = cms.vstring('HLT_HIJet35U'),
    andOr = cms.bool(True)
    )

# photon trigger
hltPhoton = hltHighLevel.clone(
    HLTPaths = cms.vstring('HLT_HIPhoton15'),
    andOr = cms.bool(True)
    )



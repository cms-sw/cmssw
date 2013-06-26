from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import * 
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import * 

L1T1coll = hltLevel1GTSeed.clone()
L1T1coll.L1TechTriggerSeeding = cms.bool(True)
L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')


from DPGAnalysis.Skims.goodvertexSkim_cff import *

selectHF = cms.EDFilter("SelectHFMinBias",
applyfilter = cms.untracked.bool(True)
)

goodcollL1requirement = cms.Sequence(L1T1coll)
goodcollHFrequirement = cms.Sequence(selectHF)



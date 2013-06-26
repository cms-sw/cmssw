from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1RequestTecAlgos = hltLevel1GTSeed.clone()

l1RequestTecAlgos.L1TechTriggerSeeding = cms.bool(True)
l1RequestTecAlgos.L1SeedsLogicalExpression = cms.string('31')

rpcTecSkimseq = cms.Sequence(l1RequestTecAlgos)

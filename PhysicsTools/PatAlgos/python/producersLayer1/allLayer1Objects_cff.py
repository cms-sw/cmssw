import FWCore.ParameterSet.Config as cms

# Build the Objects from AOD (Jets, Muons, Electrons, METs, Taus)
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.mhtProducer_cff import *

# One module to count objects
allLayer1Summary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("allLayer1Objects|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("allLayer1Electrons"),
        cms.InputTag("allLayer1Muons"),
        cms.InputTag("allLayer1Taus"),
        cms.InputTag("allLayer1Photons"),
        cms.InputTag("allLayer1Jets"),
        cms.InputTag("layer1METs"),
        cms.InputTag("layer1MHTs")        
    )
)

allLayer1Objects = cms.Sequence(
    makeAllLayer1Electrons +
    makeAllLayer1Muons +
    makeAllLayer1Taus +
    makeAllLayer1Photons +
    makeAllLayer1Jets +
    makeLayer1METs +
    makeLayer1MHTs +    
    allLayer1Summary
)

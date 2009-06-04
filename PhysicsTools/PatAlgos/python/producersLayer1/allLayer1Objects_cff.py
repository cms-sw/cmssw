import FWCore.ParameterSet.Config as cms

# Build the Objects from AOD (Jets, Muons, Electrons, METs, Taus)
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import *

# One module to count objects
allLayer1Summary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("allLayer1Objects|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("allLayer1Electrons"),
        cms.InputTag("allLayer1Muons"),
        cms.InputTag("allLayer1Taus"),
        cms.InputTag("allLayer1Photons"),
        cms.InputTag("allLayer1Jets"),
    )
)

allLayer1Objects = cms.Sequence(
    allLayer1Electrons +
    allLayer1Muons +
    allLayer1Taus +
    allLayer1Photons +
    allLayer1Jets +
    layer1METs +
    allLayer1Summary
)
allLayer1Objects.doc = "Produce PAT objects, without any selection"



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
patCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("patCandidates|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("patElectrons"),
        cms.InputTag("patMuons"),
        cms.InputTag("patTaus"),
        cms.InputTag("patPhotons"),
        cms.InputTag("patJets"),
        cms.InputTag("patMETs"),
#       cms.InputTag("patMHTs")
    )
)

patCandidates = cms.Sequence(
    makePatElectrons +
    makePatMuons     +
    makePatTaus      +
    makePatPhotons   +
    makePatJets      +
    makePatMETs      +
#   makePatMHTs      +    
    patCandidateSummary
)

import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cff import *

## module to count objects
patCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("patCandidates|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("patElectrons"),
        cms.InputTag("patMuons"),
        cms.InputTag("patTaus"),
        cms.InputTag("patPhotons"),
        cms.InputTag("patJets"),
        cms.InputTag("patMETs"),
    )
)

# PU SUB AND PARTICLES FOR ISO ---------------

from CommonTools.ParticleFlow.pfNoPileUp_cff import *
from CommonTools.ParticleFlow.pfParticleSelection_cff import *

# note pfPileUp modified according to JetMET's recommendations
pfPileUp.checkClosestZVertex = False
pfPileUp.Vertices = 'goodOfflinePrimaryVertices'
pfPileUp.PFCandidates = 'particleFlow'
pfNoPileUp.bottomCollection = 'particleFlow'

from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
pfNoPileUpSequence.insert(0, goodOfflinePrimaryVertices)

PATPileUpSubtractionSequence = cms.Sequence(
    pfNoPileUpSequence +
    pfParticleSelectionSequence
    )

## for scheduled mode
patCandidates = cms.Sequence(
    PATPileUpSubtractionSequence +
    makePatElectrons +
    makePatMuons     +
    makePatTaus      +
    makePatPhotons   +
    makePatJets      +
    makePatMETs      +
    patCandidateSummary
)


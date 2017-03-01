import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
genMetCaloAndNonPrompt = cms.EDProducer(
    "GenMETProducer",
    src = cms.InputTag("genParticlesForJets"),
    alias = cms.string('genMetCaloAndNonPrompt'), ## Alias for FWLite
    onlyFiducialParticles = cms.bool(True), ## use only fiducial GenParticles
    globalThreshold = cms.double(0.0), ## Global Threshold for input objects
    usePt = cms.bool(True), ## using Pt instead Et
    applyFiducialThresholdForFractions  = cms.bool(False),

)

##____________________________________________________________________________||

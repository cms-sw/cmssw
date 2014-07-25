import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
genMetTrue = cms.EDProducer(
    "GenMETProducer",
    src = cms.InputTag("genParticlesForMETAllVisible"),
    alias = cms.string('genMetTrue'), ## Alias for FWLite
    onlyFiducialParticles = cms.bool(False), ## Use only fiducial GenParticles
    globalThreshold = cms.double(0.0), ## Global Threshold for input objects
    usePt   = cms.bool(True), ## using Pt instead Et
    applyFiducialThresholdForFractions   = cms.bool(False),
    )

##____________________________________________________________________________||

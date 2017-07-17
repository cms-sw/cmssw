import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
genMetCalo = cms.EDProducer(
    "GenMETProducer",
    src = cms.InputTag("genCandidatesForMET"),
    alias = cms.string('genMetCalo'), ## Alias for FWLite
    onlyFiducialParticles = cms.bool(True), ## Use Only Fiducial Gen Particles
    globalThreshold = cms.double(0.0), ## Global Threshold for input objects
    usePt   = cms.bool(True), ## using Pt instead Et
    applyFiducialThresholdForFractions   = cms.bool(False),
    )

##____________________________________________________________________________||


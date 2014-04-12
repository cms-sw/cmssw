import FWCore.ParameterSet.Config as cms

# correct computation of the GenMET, with muons

genParticlesForMETAllVisible = cms.EDFilter(
    "InputGenJetsParticleSelector",
    src = cms.InputTag("genParticles"),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(False),
    excludeFromResonancePids = cms.vuint32(),
    tausAsJets = cms.bool(False),
    
    ignoreParticleIDs = cms.vuint32(
    1000022, 2000012, 2000014,
    2000016, 1000039, 5000039,
    4000012, 9900012, 9900014,
    9900016, 39, 12, 14, 16
    )
    )                   

genMetTrue = cms.EDProducer("GenMETProducer",
    src = cms.InputTag("genParticlesForMETAllVisible"), ## Input  product label       
    alias = cms.string('GenMETAllVisible'), ## Alias  for FWLite          
    onlyFiducialParticles = cms.bool(False), ## Use only fiducial GenParticles
    globalThreshold = cms.double(0.0), ## Global Threshold for input objects
    usePt   = cms.bool(True), ## using Pt instead Et
    applyFiducialThresholdForFractions   = cms.bool(False),
)

genMetTrueSequence = cms.Sequence(
    genParticlesForMETAllVisible *
    genMetTrue
    )

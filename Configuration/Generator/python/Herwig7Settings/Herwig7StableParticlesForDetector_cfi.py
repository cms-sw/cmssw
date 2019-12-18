import FWCore.ParameterSet.Config as cms

herwig7StableParticlesForDetectorBlock = cms.PSet(
    herwig7StableParticlesForDetector = cms.vstring(
        'set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm',
        'set /Herwig/Decays/DecayHandler:LifeTimeOption Average',
        )
)

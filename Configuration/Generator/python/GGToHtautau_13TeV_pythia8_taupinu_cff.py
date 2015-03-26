import FWCore.ParameterSet.Config as cms



generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13000.0),
                         crossSection = cms.untracked.double(6.44),
                         filterEfficiency = cms.untracked.double(1),
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         PythiaParameters = cms.PSet(
    processParameters = cms.vstring(
    'Main:timesAllowErrors = 10000',
    'ParticleDecays:limitTau0 = on',
    'ParticleDecays:tauMax = 10',
    'Tune:ee 3',
    'Tune:pp 5',
    'HiggsSM:gg2H = on',
    '25:onMode = off',
    '25:onIfAny = 15',
    '25:mMin = 50.',
    '15:onMode = off',
    '15:onIfMatch = 16 -211'
    ),
    parameterSets = cms.vstring('processParameters')
    )
                         )

ProductionFilterSequence = cms.Sequence(generator)

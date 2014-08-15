import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

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
    
    'WeakSingleBoson:ffbar2W = on',
    '24:onMode = off',
    '24:onIfAny = 11,12,13,14,15,16',
    '24:mMin = 50.',
    
    ),
    parameterSets = cms.vstring('processParameters')
    )
                         )

ProductionFilterSequence = cms.Sequence(generator)

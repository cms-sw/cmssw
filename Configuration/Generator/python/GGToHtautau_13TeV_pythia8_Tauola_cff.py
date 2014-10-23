import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         comEnergy = cms.double(13000.0),
                         crossSection = cms.untracked.double(6.44),
                         filterEfficiency = cms.untracked.double(1),
                         maxEventsToPrint = cms.untracked.int32(1),
                         ExternalDecays = cms.PSet(
    Tauola = cms.untracked.PSet(
    UseTauolaPolarization = cms.bool(True),
    InputCards = cms.PSet(
    mdtau = cms.int32(0),
    pjak2 = cms.int32(0),
    pjak1 = cms.int32(0)
    )
    ),
    parameterSets = cms.vstring('Tauola')
    ),
                         UseExternalGenerators = cms.untracked.bool(True),
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
    ),
    parameterSets = cms.vstring('processParameters')
    )
                         )

ProductionFilterSequence = cms.Sequence(generator)

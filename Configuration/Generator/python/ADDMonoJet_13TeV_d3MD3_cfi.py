import FWCore.ParameterSet.Config as cms
generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythia8_unparticle = cms.vstring(
        'Tune:pp = 5',
        'PDF:pSet = 5',
        'ExtraDimensionsLED:monojet = on',
        'ExtraDimensionsLED:CutOffmode = 1',
	'ExtraDimensionsLED:t = 0.5',
        'ExtraDimensionsLED:n = 3',
        'ExtraDimensionsLED:MD = 3000.',
        '5000039:m0 = 1200.',
        '5000039:mWidth = 1000.',
        '5000039:mMin = 1.',
        '5000039:mMax = 13990.',
        'PhaseSpace:pTHatMin = 80.',
        'PartonLevel:MI = on',
        'PartonLevel:ISR = on',
        'PartonLevel:FSR = on',
        'ParticleDecays:limitTau0 = on',
        'ParticleDecays:tauMax = 10'
	),
        parameterSets = cms.vstring('pythia8_unparticle')
    )
)

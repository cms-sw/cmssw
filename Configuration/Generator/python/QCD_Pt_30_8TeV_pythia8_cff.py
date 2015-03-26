import FWCore.ParameterSet.Config as cms



generator = cms.EDFilter("Pythia8GeneratorFilter",
    crossSection = cms.untracked.double(5.72e+07),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(8000.0),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
	    'Main:timesAllowErrors    = 10000', 
            'ParticleDecays:limitTau0 = on',
	    'ParticleDecays:tauMax = 10',
            'HardQCD:all = on',
            'PhaseSpace:pTHatMin = 30.',
            'Tune:pp 2',                      
            'Tune:ee 3'),
        parameterSets = cms.vstring('processParameters')
    )
)

ProductionFilterSequence = cms.Sequence(generator)

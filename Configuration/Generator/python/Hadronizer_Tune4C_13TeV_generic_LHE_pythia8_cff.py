import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
	    'Main:timesAllowErrors    = 10000', 
        'ParticleDecays:limitTau0 = on',
	    'ParticleDecays:tauMax = 10',
        'Tune:ee 3',
        'Tune:pp 5'),
        parameterSets = cms.vstring('processParameters')
    )
)

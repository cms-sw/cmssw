import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8GeneratorFilter",
    comEnergy = cms.double(10000.),

    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),

    filterEfficiency = cms.untracked.double(1.0),

    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring(
            'pythia8_example01'
        ),
        pythia8_example01 = cms.vstring(
            'HardQCD:all = on',
            'PhaseSpace:pTHatMin = 20.'
        )
    )
)

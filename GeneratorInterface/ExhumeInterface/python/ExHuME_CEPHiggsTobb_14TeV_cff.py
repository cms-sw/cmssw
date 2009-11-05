import FWCore.ParameterSet.Config as cms

from GeneratorInterface.ExhumeInterface.ExhumeParameters_cfi import ExhumeParameters as ExhumeParametersRef

generator = cms.EDFilter("ExhumeGeneratorFilter",
    PythiaParameters = cms.PSet(
       parameterSets = cms.vstring()
    ),
    ExhumeParameters = ExhumeParametersRef,
    comEnergy = cms.double(14000.),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    ExhumeProcess = cms.PSet(
        ThetaMin = cms.double(0.95),
        MassRangeLow = cms.double(115.0),
        MassRangeHigh = cms.double(125.0),
        ProcessType = cms.string('Higgs'),
        HiggsDecay = cms.int32(5),
    )
)

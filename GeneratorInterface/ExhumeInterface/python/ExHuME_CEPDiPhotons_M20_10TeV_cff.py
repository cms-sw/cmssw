import FWCore.ParameterSet.Config as cms

from GeneratorInterface.ExhumeInterface.ExhumeParameters_cfi import ExhumeParameters as ExhumeParametersRef

source = cms.Source("EmptySource")

generator = cms.EDFilter("ExhumeGeneratorFilter",
    PythiaParameters = cms.PSet(
       parameterSets = cms.vstring()
    ),
    ExhumeParameters = ExhumeParametersRef,
    comEnergy = cms.double(10000.),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    ExhumeProcess = cms.PSet(
       ProcessType = cms.string('DiPhoton'),
       ThetaMin = cms.double(0.30),
       MassRangeLow = cms.double(20.0),
       MassRangeHigh = cms.double(9999.0)
    )
)

ProductionFilterSequence = cms.Sequence(generator)

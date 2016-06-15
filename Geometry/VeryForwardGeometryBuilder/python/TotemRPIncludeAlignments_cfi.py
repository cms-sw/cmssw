import FWCore.ParameterSet.Config as cms

TotemRPIncludeAlignments = cms.ESSource("TotemRPIncludeAlignments",
    verbosity = cms.untracked.uint32(1),

    MeasuredFiles = cms.vstring(),
    RealFiles = cms.vstring(),
    MisalignedFiles = cms.vstring()
)

import FWCore.ParameterSet.Config as cms

ctppsIncludeAlignments = cms.ESSource("CTPPSIncludeAlignments",
    verbosity = cms.untracked.uint32(0),

    MeasuredFiles = cms.vstring(),
    RealFiles = cms.vstring(),
    MisalignedFiles = cms.vstring()
)

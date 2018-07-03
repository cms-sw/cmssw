import FWCore.ParameterSet.Config as cms

ctppsIncludeAlignmentsFromXML = cms.ESSource("CTPPSIncludeAlignmentsFromXML",
    verbosity = cms.untracked.uint32(0),

    MeasuredFiles = cms.vstring(),
    RealFiles = cms.vstring(),
    MisalignedFiles = cms.vstring()
)

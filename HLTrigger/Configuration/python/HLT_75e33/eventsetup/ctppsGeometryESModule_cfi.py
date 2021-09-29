import FWCore.ParameterSet.Config as cms

ctppsGeometryESModule = cms.ESProducer("CTPPSGeometryESModule",
    compactViewTag = cms.string('XMLIdealGeometryESSource_CTPPS'),
    verbosity = cms.untracked.uint32(1)
)

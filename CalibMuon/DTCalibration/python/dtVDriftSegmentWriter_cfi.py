import FWCore.ParameterSet.Config as cms

dtVDriftSegmentWriter = cms.EDAnalyzer("DTVDriftWriter",
    vDriftAlgo = cms.string('DTVDriftSegment'),
    vDriftAlgoConfig = cms.PSet(
        rootFileName = cms.string(''),
        nSigmasFitRange = cms.untracked.uint32(1),
        debug = cms.untracked.bool(False)
    )
)

import FWCore.ParameterSet.Config as cms

dtVDriftSegmentWriter = cms.EDAnalyzer("DTVDriftWriter",
    vDriftAlgo = cms.string('DTVDriftSegment'),
    readLegacyVDriftDB =cms.bool(True),
    writeLegacyVDriftDB =cms.bool(True),
    vDriftAlgoConfig = cms.PSet(
        rootFileName = cms.string(''),
        nSigmasFitRange = cms.untracked.uint32(1),
        readLegacyVDriftDB =cms.bool(True),
        debug = cms.untracked.bool(False)
    )
)

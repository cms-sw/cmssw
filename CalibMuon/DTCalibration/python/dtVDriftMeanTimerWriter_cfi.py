import FWCore.ParameterSet.Config as cms

dtVDriftMeanTimerWriter = cms.EDAnalyzer("DTVDriftWriter",
    vDriftAlgo = cms.string('DTVDriftMeanTimer'),
    vDriftAlgoConfig = cms.PSet(
        rootFileName = cms.string(''),
        debug = cms.untracked.bool(False)
    )
)

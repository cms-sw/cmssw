import FWCore.ParameterSet.Config as cms

dtVDriftMeanTimerWriter = cms.EDAnalyzer("DTVDriftWriter",
    vDriftAlgo = cms.string('DTVDriftMeanTimer'),
    vDriftAlgoConfig = cms.PSet(
        rootFileName = cms.string(''),
        debug = cms.untracked.bool(False)
    ),
    readLegacyVDriftDB = cms.bool(True),
    writeLegacyVDriftDB = cms.bool(True)
)
# foo bar baz
# 1eSemKq4nbLOk
# IMjGL2i9g4wg0

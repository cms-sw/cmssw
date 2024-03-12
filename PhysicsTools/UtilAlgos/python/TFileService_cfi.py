import FWCore.ParameterSet.Config as cms

TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root'),
    closeFileFast = cms.untracked.bool(False)
)
# foo bar baz
# 3Mya7hBXHR1QZ

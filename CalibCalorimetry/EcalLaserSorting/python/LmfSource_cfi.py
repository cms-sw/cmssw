import FWCore.ParameterSet.Config as cms

source = cms.Source("LmfSource",
  fileNames = cms.vstring("in.lmf"),
  preScale  = cms.uint32(1),
  orderedRead = cms.bool(True),
  watchFileList = cms.bool(False),
  fileListName = cms.string("fileList.txt"),
  inputDir = cms.string(""),
  nSecondsToSleep = cms.int32(5),
  verbosity = cms.untracked.int32(0)
)
# foo bar baz
# DiPQMjZQlfGuI
# Zs3WyBjfB9cT7

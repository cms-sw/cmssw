import FWCore.ParameterSet.Config as cms

l1tdttfClient = cms.EDAnalyzer("L1TDTTFClient",
    l1tSourceFolder = cms.untracked.string('L1T/L1TDTTF'),
    dttfSource = cms.InputTag("l1tdttf"),
    online = cms.untracked.bool(True),
    resetAfterLumi = cms.untracked.int32(3)
)



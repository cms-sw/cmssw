import FWCore.ParameterSet.Config as cms

l1tGctClient = cms.EDAnalyzer("L1TGCTClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string('L1T/L1TGCT'),
    prescaleEvt = cms.untracked.int32(1),
    runInEventLoop=cms.untracked.bool(False),
    runInEndLumi=cms.untracked.bool(True),
    runInEndRun=cms.untracked.bool(True),
    runInEndJob=cms.untracked.bool(False),
    stage1_layer2_=cms.untracked.bool(False)
)



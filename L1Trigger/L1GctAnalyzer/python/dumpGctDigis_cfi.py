import FWCore.ParameterSet.Config as cms

dumpGctDigis = cms.EDAnalyzer("DumpGctDigis",
    doRctEm = cms.untracked.bool(True),
    doJets = cms.untracked.bool(False),
    doEmulated = cms.untracked.bool(False),
    emuGctInput = cms.untracked.InputTag("gctDigis"),
    doRegions = cms.untracked.bool(False),
    doEm = cms.untracked.bool(False),
    outFile = cms.untracked.string('gctDigis.txt'),
    doInternEm = cms.untracked.bool(True),
    emuRctInput = cms.untracked.InputTag("rctDigis"),
    rawInput = cms.untracked.InputTag("l1GctHwDigis"),
    doEnergySums = cms.untracked.bool(False),
    doFibres = cms.untracked.bool(True),
    doHardware = cms.untracked.bool(True),
    emMinRank = cms.untracked.uint32(1),
    jetMinRank = cms.untracked.uint32(1)
)



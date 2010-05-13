import FWCore.ParameterSet.Config as cms

gctErrorAnalyzer = cms.EDAnalyzer('GctErrorAnalyzer',
    #Multiple BX Flags
    doRCTMBx = cms.untracked.bool(False),
    doEmuMBx = cms.untracked.bool(False),
    doGCTMBx = cms.untracked.bool(False),
    #Plot + Debug Info Flags
    doRCT = cms.untracked.bool(True),
    doEg = cms.untracked.bool(True),
    doIsoDebug = cms.untracked.bool(True),
    doNonIsoDebug = cms.untracked.bool(True),
    doJets = cms.untracked.bool(True),
    doCenJetsDebug = cms.untracked.bool(True),
    doTauJetsDebug = cms.untracked.bool(True),
    doForJetsDebug = cms.untracked.bool(True),
    doHF = cms.untracked.bool(True),
    doRingSumDebug = cms.untracked.bool(True),
    doBitCountDebug = cms.untracked.bool(True),
    doTotalEnergySums = cms.untracked.bool(True),
    doTotalHtDebug = cms.untracked.bool(True),
    doTotalEtDebug = cms.untracked.bool(True),
    doMissingEnergySums = cms.untracked.bool(True),
    doMissingETDebug = cms.untracked.bool(True),
    doMissingHTDebug = cms.untracked.bool(True),
    doExtraMissingHTDebug = cms.untracked.bool(False),
    #Labels to use for data and emulator
    dataTag = cms.untracked.InputTag("l1GctHwDigis"),
    emuTag = cms.untracked.InputTag("valGctDigis"),
    #Nominally, the following parameters should NOT be changed
    RCTTrigBx = cms.untracked.int32(0),
    EmuTrigBx = cms.untracked.int32(0),
    GCTTrigBx = cms.untracked.int32(0),
    #Choose the Geometry of the system
    useSys = cms.untracked.string("P5")          
)


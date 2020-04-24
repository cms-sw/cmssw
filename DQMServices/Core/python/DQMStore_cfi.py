import FWCore.ParameterSet.Config as cms

DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    verboseQT = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False),
    enableMultiThread = cms.untracked.bool(False),
    #the LSbasedMode flag is needed for the online. All the
    #MEs are flagged to be LS based.
    LSbasedMode = cms.untracked.bool(False),
    #this is bound to the enableMultiThread flag.
    forceResetOnBeginLumi = cms.untracked.bool(False)
)

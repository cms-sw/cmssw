import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

triggerSynchTest = DQMEDHarvester("DTLocalTriggerSynchTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('TM'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),                         
    # root folder for booking of histograms
    folderRoot = cms.untracked.string(''),
    # correlated fraction test tresholds
    bxTimeInterval  = cms.double(25),
    rangeWithinBX   = cms.bool(True),
    nBXHigh         = cms.int32(0),
    nBXLow          = cms.int32(1),
    minEntries      = cms.int32(200),
    writeDB         = cms.bool(True),
    dbFromTM       = cms.bool(False),
    fineParamDiff   = cms.bool(False),
    coarseParamDiff = cms.bool(False),
    numHistoTag     = cms.string('TrackCrossingTimeAllInBX'),
    denHistoTag     = cms.string('TrackCrossingTimeHHInBX'),
    ratioHistoTag   = cms.string('TrackCrossingTimeAllOverHHInBX')                                  
)



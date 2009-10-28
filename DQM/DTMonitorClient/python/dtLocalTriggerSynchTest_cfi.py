import FWCore.ParameterSet.Config as cms

triggerSynchTest = cms.EDAnalyzer("DTLocalTriggerSynchTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('DCC','DDU'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),                         
    # root folder for booking of histograms
    folderRoot = cms.untracked.string(''),
    # correlated fraction test tresholds
    bxTimeInterval  = cms.double(25),
    rangeWithinBX   = cms.bool(True),
    writeDB         = cms.bool(True),
    dbFromDCC       = cms.bool(False),
    fineParamDiff   = cms.bool(False),
    coarseParamDiff = cms.bool(False),
    numHistoTag     = cms.string("TrackCrossingTimeAll"),
    denHistoTag     = cms.string("TrackCrossingTimeHH"),
    ratioHistoTag   = cms.string("TrackCrossingTimeAllOverHH")                                  
)



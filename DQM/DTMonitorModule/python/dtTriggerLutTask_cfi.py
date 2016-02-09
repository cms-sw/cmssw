import FWCore.ParameterSet.Config as cms

dtTriggerLutMonitor = cms.EDAnalyzer("DTLocalTriggerLutTask",
    # labels of DDU/TM data and 4D segments
    inputTagTM = cms.untracked.InputTag("twinMuxStage2Digis"),
    inputTagSEG = cms.untracked.InputTag("dt4DSegments"),
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    # set outflows to boudaries
    rebinOutFlowsInGraph = cms.untracked.bool(True),
    # enable more detailed studies
    detailedAnalysis = cms.untracked.bool(False),
    # label of the geometry used to feed DTTrigGeomUtils
    geomLabel = cms.untracked.string("idealForDigi"),
    # number of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(9999)
)



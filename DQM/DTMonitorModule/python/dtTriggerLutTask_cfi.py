import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtTriggerLutMonitor = DQMEDAnalyzer('DTLocalTriggerLutTask',
    # labels of TM data and 4D segments
    inputTagTMin = cms.untracked.InputTag('twinMuxStage2Digis:PhIn'),
    inputTagTMout = cms.untracked.InputTag('twinMuxStage2Digis:PhOut'),
    inputTagSEG = cms.untracked.InputTag('dt4DSegments'),
    # set static booking (all the detector)
    staticBooking = cms.untracked.bool(True),
    # set outflows to boudaries
    rebinOutFlowsInGraph = cms.untracked.bool(True),
    # enable more detailed studies
    detailedAnalysis = cms.untracked.bool(False),
    # label of the geometry used to feed DTTrigGeomUtils
    geomLabel = cms.untracked.string('idealForDigi'),
    # number of luminosity blocks to reset the histos
    ResetCycle = cms.untracked.int32(9999)
)



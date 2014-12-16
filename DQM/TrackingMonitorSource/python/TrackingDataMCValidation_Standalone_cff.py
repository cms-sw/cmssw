import FWCore.ParameterSet.Config as cms
from DQM.TrackingMonitorSource.StandaloneTrackMonitor_cfi import *

# Primary Vertex Selector
selectedPrimaryVertices = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("!isFake && ndof >= 4 && abs(z) < 24 && abs(position.Rho) < 2.0"),
    filter = cms.bool(False)
)
# Track Selector
selectedTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag('generalTracks'),
    cut = cms.string("pt > 0.5"),
    filter = cms.bool(False)
)
hltPathFilter = cms.EDFilter("HLTPathFilter",
    processName = cms.string("HLT"),
    triggerName = cms.string("HLT_ZeroBias_v7"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")
)
ztoMMEventSelector = cms.EDFilter("ZtoMMEventSelector")
ztoEEEventSelector = cms.EDFilter("ZtoEEEventSelector")

standaloneTrackMonitorEE = standaloneTrackMonitor.clone()
standaloneValidationElec = cms.Sequence(
                                 selectedPrimaryVertices
                               * selectedTracks
                               * ztoEEEventSelector
                               * standaloneTrackMonitorEE)

standaloneTrackMonitorMM = standaloneTrackMonitor.clone()
standaloneValidationMuon = cms.Sequence(
                                 selectedPrimaryVertices
                               * selectedTracks
                               * ztoMMEventSelector
                               * standaloneTrackMonitorMM)

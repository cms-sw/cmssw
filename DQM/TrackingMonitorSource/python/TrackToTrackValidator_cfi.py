import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.histoHelper4hltTracking_cfi import *
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

trackToTrackValidator = DQMEDAnalyzer("TrackToTrackValidator",
    monitoredTrack           = cms.InputTag("hltMergedTracks"),
    referenceTrack           = cms.InputTag("generalTracks"),
    monitoredBeamSpot        = cms.InputTag("hltOnlineBeamSpot"),
    referenceBeamSpot        = cms.InputTag("offlineBeamSpot"),
    monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector"),
    referencePrimaryVertices = cms.InputTag("offlinePrimaryVertices"),
    topDirName         = cms.string("HLT/Tracking/ValidationWRTreco"),

    dRmin         = cms.double(0.002),

    # HistoProducerAlgo. Defines the set of plots to be booked and filled
    histoPSet = histoHelper4hltTracking,

)

import FWCore.ParameterSet.Config as cms

#
# Analyzers
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")

TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)

plotEvent = cms.EDFilter("EventPlotter",
    trackCollection = cms.vstring('globalPrimTracks', 'globalSecoTracks'),
    zipFiles = cms.bool(False)
)

testPixel3Tracks = cms.EDAnalyzer("LowPtTrackAnalyzer",
    trackCollection = cms.vstring('pixel3PrimTracks'),
    zipFiles = cms.bool(False),
    plotEvent = cms.bool(False),
    resultFile = cms.string('resultPixel3Tracks.root')
)

testGlobalTracks = cms.EDAnalyzer("LowPtTrackAnalyzer",
    trackCollection = cms.vstring('globalPrimTracks'),
    zipFiles = cms.bool(False),
    plotEvent = cms.bool(False),
    resultFile = cms.string('resultGlobalTracks.root')
)

multiplicity = cms.EDAnalyzer("ChargedMultiplicityAnalyzer")

TrackAssociatorByHitsESProducer.MinHitCut = 0.0


import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonPixelTrackCleanerBySharedHits = cms.ESProducer("PixelTrackCleanerBySharedHitsESProducer",
    ComponentName = cms.string('hltPhase2L3MuonPixelTrackCleanerBySharedHits'),
    appendToDataLabel = cms.string(''),
    useQuadrupletAlgo = cms.bool(False)
)

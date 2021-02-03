import FWCore.ParameterSet.Config as cms

TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer",
    trackerGeometryLabel = cms.untracked.string('')
)

import FWCore.ParameterSet.Config as cms
def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def customiseStoreTrajectoryInEvent(process):
    for producer in producers_by_type(process,"TrackProducer"):
       producer.TrajectoryInEvent = cms.bool(True)
       producer.useHitsSplitting = cms.bool(False)  # HI still set this on...
    for producer in producers_by_type(process,"TrackListMerger"):
       producer.copyExtras = cms.untracked.bool(True)
       producer.copyTrajectories = cms.untracked.bool(True)
    for producer in producers_by_type(process,"TrackCollectionMerger"):
       producer.copyExtras = cms.untracked.bool(True)
       producer.copyTrajectories = cms.untracked.bool(True)
    for producer in producers_by_type(process,"DuplicateListMerger"):
       producer.copyExtras = cms.untracked.bool(True)
       producer.copyTrajectories = cms.untracked.bool(True)
    for producer in producers_by_type(process,"TrackCollectionFilterCloner"):
       producer.copyExtras = cms.untracked.bool(True)
       producer.copyTrajectories = cms.untracked.bool(True)
    for producer in producers_by_type(process,"AnalyticalTrackSelector") :
       producer.copyExtras = cms.untracked.bool(True)
       producer.copyTrajectories = cms.untracked.bool(True)
    return process

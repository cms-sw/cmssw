import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the Tracker geometry model.
#
TrackerDigiGeometryESModule = cms.ESProducer("TrackerDigiGeometryESModule",
    fromDDD = cms.bool(True),
    applyAlignment = cms.untracked.bool(False)
)



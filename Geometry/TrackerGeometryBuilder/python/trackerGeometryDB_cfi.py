import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the Tracker geometry model.
#
# GF would like to have a shorter name (e.g. TrackerGeometry), but since originally
# there was no name, replace statements in other configs would not work anymore...
TrackerDigiGeometryESModule = cms.ESProducer("TrackerDigiGeometryESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(False),
    applyAlignment = cms.bool(True), # to be abondoned
    applySurfaceDeformation = cms.bool(True),

    alignmentsLabel = cms.string('')
)



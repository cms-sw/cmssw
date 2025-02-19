import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the DT geometry model.
#
# GF would like to have a shorter name (e.g. DTGeometry), but since originally
# there was no name, replace statements in other configs would not work anymore...
DTGeometryESModule = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(True), ## to be abondoned (?)

    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(False)
)



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

    alignmentsLabel = cms.string(''),
    upgradeGeometry = cms.untracked.bool(True),
    ROWS_PER_ROC = cms.untracked.int32(80),
    COLS_PER_ROC = cms.untracked.int32(52),
    BIG_PIX_PER_ROC_X = cms.untracked.int32(0),
    BIG_PIX_PER_ROC_Y = cms.untracked.int32(0),
    ROCS_X = cms.untracked.int32(2),
    ROCS_Y = cms.untracked.int32(8)
)



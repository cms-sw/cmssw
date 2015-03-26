import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the CSC geometry model.
#
# modelling flags (for completeness - internal defaults are already sane)
# GF would like to have a shorter name (e.g. CSCGeometry), but since originally
# there was no name, replace statements in other configs would not work anymore...
CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string(''),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(True), ## GF: to be abandoned
    useDDD = cms.bool(False)
)

#
# Modify for running in run 2
#
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify( CSCGeometryESModule, useGangedStripsInME1a=False )

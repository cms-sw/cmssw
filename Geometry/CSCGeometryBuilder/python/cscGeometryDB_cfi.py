import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the CSC geometry model.
#
# modelling flags (for completeness - internal defaults are already sane)
# GF would like to have a shorter name (e.g. CSCGeometry), but since originally
# there was no name, replace statements in other configs would not work anymore...
CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
  fromDDD = cms.bool(False),
  fromDD4hep = cms.bool(False),
  alignmentsLabel = cms.string(''),
  appendToDataLabel = cms.string(''),
  useRealWireGeometry = cms.bool(True),
  useOnlyWiresInME1a = cms.bool(False),
  useGangedStripsInME1a = cms.bool(True),
  useCentreTIOffsets = cms.bool(False),
  applyAlignment = cms.bool(True),  ## GF: to be abandoned
  debugV = cms.untracked.bool(False)
)

#
# Modify for running in run 2
#
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( CSCGeometryESModule, useGangedStripsInME1a=False )

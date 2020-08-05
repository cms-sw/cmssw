import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the CSC geometry model.
#
# modelling flags (for completeness - internal defaults are already sane)
# GF would like to have a shorter name (e.g. CSCGeometry), but since originally
# there was no name, replace statements in other configs would not work anymore...

from Geometry.CSCGeometryBuilder.CSCGeometryESModule_cfi import CSCGeometryESModule

#
# Modify for running in run 2
#
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( CSCGeometryESModule, useGangedStripsInME1a = False )

#
# Modify for running with dd4hep
#
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify( CSCGeometryESModule, fromDDD = False, fromDD4hep = True )

import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the DT geometry model.
#
# GF would like to have a shorter name (e.g. DTGeometry), but since originally
# there was no name, replace statements in other configs would not work anymore...

from Geometry.DTGeometryBuilder.DTGeometryESModule_cfi import DTGeometryESModule

#
# Modify for running with dd4hep
#
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify( DTGeometryESModule, fromDDD = False, fromDD4hep = True )


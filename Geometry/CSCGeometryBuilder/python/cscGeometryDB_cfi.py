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

dd4hep.toModify( CSCGeometryESModule, useDDD = False, useDD4hep = True )

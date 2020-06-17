from Geometry.RPCGeometryBuilder.RPCGeometryESModule_cfi import RPCGeometryESModule

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(RPCGeometryESModule, useDDD = False, useDD4hep = True)

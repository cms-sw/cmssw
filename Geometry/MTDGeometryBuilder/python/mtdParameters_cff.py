from Geometry.MTDGeometryBuilder.mtdParameters_cfi import mtdParameters

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(mtdParameters, fromDD4hep = True)

from Geometry.ForwardGeometry.zdcTopologyEP_cfi import *
from Geometry.ForwardGeometry.zdcHardcodeGeometryEP_cfi import zdcHardcodeGeometryEP
from Configuration.Eras.Modifier_zdcAddRPD_cff import zdcAddRPD

zdcAddRPD.toModify(zdcHardcodeGeometryEP, zdcAddRPD = True)

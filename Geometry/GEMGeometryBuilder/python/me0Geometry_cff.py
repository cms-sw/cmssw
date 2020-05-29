from Geometry.GEMGeometryBuilder.me0Geometry_cfi import me0Geometry

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(me0Geometry, fromDDD = False, fromDD4hep = True)

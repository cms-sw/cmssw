from Geometry.GEMGeometryBuilder.gemGeometry_cfi import gemGeometry

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(gemGeometry, fromDDD = False, fromDD4Hep = True)

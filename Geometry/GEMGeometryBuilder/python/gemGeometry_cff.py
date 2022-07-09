from Geometry.GEMGeometryBuilder.gemGeometry_cfi import gemGeometry

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

dd4hep.toModify(gemGeometry, fromDDD = False, fromDD4hep = True)

run3_GEM.toModify(gemGeometry, applyAlignment = True)

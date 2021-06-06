from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi import mtdNumberingGeometry

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(mtdNumberingGeometry, fromDDD = False, fromDD4hep = True)

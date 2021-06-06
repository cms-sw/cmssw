from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import trackerNumberingGeometry

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(trackerNumberingGeometry, fromDDD = False, fromDD4hep = True)

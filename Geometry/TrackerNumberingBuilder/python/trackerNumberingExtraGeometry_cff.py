from Geometry.TrackerNumberingExtraBuilder.trackerNumberingExtraGeometry_cfi import trackerNumberingExtraGeometry

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(trackerNumberingExtraGeometry, fromDDD = False, fromDD4hep = True)

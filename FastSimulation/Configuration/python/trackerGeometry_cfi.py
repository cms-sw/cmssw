import FWCore.ParameterSet.Config as cms

# The aligned Tracker geometry
import Geometry.TrackerGeometryBuilder.trackerGeometry_cfi

# The same, but for a misaligned Tracker geometry
misalignedTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerGeometry_cfi.TrackerDigiGeometryESModule.clone()
# The misalignment won't be applied
misalignedTrackerGeometry.applyAlignment = False
# Label of the produced TrackerGeometry:
misalignedTrackerGeometry.appendToDataLabel = 'MisAligned'


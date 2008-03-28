import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# RoadSearchHelixMaker
from RecoTracker.FinalTrackSelectors.ctfMCTrackListCleaner_cfi import *


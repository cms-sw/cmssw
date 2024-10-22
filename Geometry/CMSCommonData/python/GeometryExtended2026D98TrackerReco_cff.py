import FWCore.ParameterSet.Config as cms

from Geometry.CMSCommonData.cmsExtendedGeometry2026D98TrackerXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff import *
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cff import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = True

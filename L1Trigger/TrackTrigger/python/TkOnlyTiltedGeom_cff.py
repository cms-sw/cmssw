################
#
#  Tracking-Only Geometry config script (Flat case)
#  
# This script is used for processing fast stub building in the tracker only geometry
# See the available scripts in the test dir
#
# GEOM is based on:
# https://github.com/cms-sw/cmssw/blob/CMSSW_9_2_X/Geometry/CMSCommonData/python/cmsExtendedGeometry2026D11XML_cfi.py
#
# S.Viret (viret_at_ipnl.in2p3.fr): 04/07/16
#
################

import FWCore.ParameterSet.Config as cms

#Tracker stuff
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = cms.bool(False)

## Here we put the xml stuff for the tracker-only geometry
#
# Need to remove the rest in order to avoid SD-related crashes in Geant4

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
       'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2019/v1/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/cmsTracker.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',   
        'Geometry/CMSCommonData/data/mgnt.xml',
        'Geometry/CMSCommonData/data/beampipe/2026/v1/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam/2026/v1/cmsBeam.xml',
        'Geometry/CMSCommonData/data/cavern.xml',
        'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixfwd.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixbar.xml',
        'Geometry/TrackerCommonData/data/trackermaterial.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/tracker.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixel.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/trackerbar.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/trackerfwd.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/trackerStructureTopology.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker4025/pixelStructureTopology.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/trackersens.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/pixelsens.xml',
        'Geometry/TrackerRecoData/data/PhaseII/TiltedTracker4025/trackerRecoMaterial.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/trackerProdCuts.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker4025/pixelProdCuts.xml',
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'
        ),
    rootNodeName = cms.string('cms:OCMS')
)




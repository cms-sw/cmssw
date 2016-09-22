################
#
#  Tracking-Only Geometry config script (Tilted case)
#  
# This script is used for processing fast stub building in the tracker only geometry
# See the available scripts in the test directory
#
# S.Viret (viret_at_ipnl.in2p3.fr): 04/07/16
#
################

import FWCore.ParameterSet.Config as cms

#Tracker stuff
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
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
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/PhaseII/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/PhaseI/cms.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',        
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/cmsTracker.xml',
        'Geometry/CMSCommonData/data/mgnt.xml',
        'Geometry/CMSCommonData/data/PhaseII/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam.xml',
        'Geometry/CMSCommonData/data/cavern.xml',
        'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
	'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdMaterials.xml',
	'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdCylinder.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwd.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdDisks.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk1.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk2.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk3.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk4.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk5.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk6.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdInnerDisk7.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk1.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk2.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk3.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk4.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk5.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk6.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk7.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk8.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk9.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdOuterDisk10.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade1.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade2.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade3.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade4.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade5.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade6.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade7.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade8.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade9.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixfwdblade10.xml',
        'Geometry/TrackerCommonData/data/PhaseI/pixbarmaterial.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarladder.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarladderfull0.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarladderfull1.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarladderfull2.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarladderfull3.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarlayer.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarlayer0.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarlayer1.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarlayer2.xml', 
        'Geometry/TrackerCommonData/data/PhaseI/pixbarlayer3.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/pixbar.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/tracker.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/trackerbar.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/trackerfwd.xml',
	'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker/trackerStructureTopology.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker/trackersens.xml',
        'Geometry/TrackerRecoData/data/PhaseII/TiltedTracker/trackerRecoMaterial.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker/trackerProdCuts.xml',
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)




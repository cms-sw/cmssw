################
#
#  Tracking-Only Geometry config script (Flat case)
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
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/cmsTracker.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',   
        'Geometry/CMSCommonData/data/mgnt.xml',
        'Geometry/CMSCommonData/data/PhaseII/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam.xml',
        'Geometry/CMSCommonData/data/cavern.xml',
        'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
	'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdMaterials.xml',
	'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdCylinder.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwd.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdDisks.xml', 
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk1.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk2.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk3.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk4.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk5.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk6.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdInnerDisk7.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk1.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk2.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk3.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk4.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk5.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk6.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk7.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk8.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk9.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdOuterDisk10.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade1.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade2.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade3.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade4.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade5.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade6.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade7.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade8.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade9.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixfwdblade10.xml',
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
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/pixbar.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/tracker.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/trackerbar.xml',
        'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/trackerfwd.xml',
	'Geometry/TrackerCommonData/data/PhaseII/FlatTracker/trackerStructureTopology.xml',
        'Geometry/TrackerSimData/data/PhaseII/FlatTracker/trackersens.xml',
        'Geometry/TrackerRecoData/data/PhaseII/FlatTracker/trackerRecoMaterial.xml',
        'Geometry/TrackerSimData/data/PhaseII/FlatTracker/trackerProdCuts.xml',
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)




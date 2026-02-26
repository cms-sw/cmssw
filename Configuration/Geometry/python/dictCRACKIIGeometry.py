maxSections = 1

commonDict = {
    "abbrev" : "O",
    "name" : "common",
    "default" : 9,
    "O10" : {
        1 : [
            'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
            'Geometry/TrackerCommonData/data/trackermaterial/2021/v2/trackermaterial.xml',
            'Geometry/CMSCommonData/data/rotations.xml',
            'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
            'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
            'Geometry/CMSCommonData/data/cms/2030/v5/cms.xml',
            #'Geometry/TrackerCommonData/data/CRack_PhaseII/cms.xml'
            'Geometry/CMSCommonData/data/cmsMother.xml',
            'Geometry/CMSCommonData/data/eta3/etaMax.xml',
            'Geometry/CMSCommonData/data/cmsTracker.xml',
            
        ],
        5 : [
            'Geometry/CMSCommonData/data/FieldParameters.xml',
        ],
        "era" : "phase2_common, phase2_trigger",
    },
}


trackerDict = {
    "abbrev" : "T",
    "name" : "tracker",
    "default" : 34,
    "TCRACK" : {
        1 : [
            'Geometry/TrackerCommonData/data/PhaseII/TFPXTEPXReordered/trackerParameters.xml',
            'Geometry/TrackerCommonData/data/trackermaterial.xml',
            'Geometry/TrackerCommonData/data/CRack_PhaseII/tracker.xml',
            'Geometry/TrackerCommonData/data/CRack_PhaseII/tob.xml',
            'Geometry/TrackerCommonData/data/CRack_PhaseII/trackersens.xml',
            'Geometry/TrackerCommonData/data/CRack_PhaseII/trackerStructureTopology.xml',

        ],
        "sim" : [
            'from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *',
            'from SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff import *',
        ],
        "reco" : [
            'from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *',
            'from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *',
            'from Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi import *',
            'from Geometry.TrackerGeometryBuilder.trackerParameters_cff import *',
            'from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *',
            'from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *',
            'trackerGeometry.applyAlignment = True',
        ],
        "era" : "phase2_tracker, phase2_3DPixels, trackingPhase2PU140",
    },
}

allDicts = [commonDict,trackerDict]

detectorVersionDict = {
   ("O10","TCRACK") : "D500",
}

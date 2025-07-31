import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generateRun4Geometry.py
# If you notice a mistake, please update the generating script, not just this config

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
        'Geometry/TrackerCommonData/data/trackermaterial/2021/v2/trackermaterial.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2030/v5/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsTracker.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TFPXTEPXReordered/trackerParameters.xml',
        'Geometry/TrackerCommonData/data/trackermaterial.xml',
        'Geometry/TrackerCommonData/data/CRack_PhaseII/tracker.xml',
        'Geometry/TrackerCommonData/data/CRack_PhaseII/tob.xml',
        'Geometry/TrackerCommonData/data/CRack_PhaseII/trackersens.xml',
        'Geometry/TrackerCommonData/data/CRack_PhaseII/trackerStructureTopology.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)

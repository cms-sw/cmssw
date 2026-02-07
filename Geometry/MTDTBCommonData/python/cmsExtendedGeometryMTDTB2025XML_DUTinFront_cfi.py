import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generateRun4Geometry.py
# If you notice a mistake, please update the generating script, not just this config

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2030/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/MTDTBCommonData/data/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsTracker.xml',
        'Geometry/MTDTBCommonData/data/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/MTDTBCommonData/data/Tracker_DD4hep_compatible/tracker.xml',
        'Geometry/MTDCommonData/data/mtdMaterial/v3/mtdMaterial.xml',
        'Geometry/MTDTBCommonData/data/btl/tb2025_DUTinFront/btl.xml',
        'Geometry/MTDCommonData/data/etl/v8/etl.xml',
        'Geometry/MTDCommonData/data/mtdParameters/v6/mtdStructureTopology.xml',
        'Geometry/MTDCommonData/data/mtdParameters/v6/mtdParameters.xml',
    )+
    cms.vstring(
        'Geometry/MTDSimData/data/v5/mtdsens.xml',
        'Geometry/MTDSimData/data/v5/mtdProdCuts.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)

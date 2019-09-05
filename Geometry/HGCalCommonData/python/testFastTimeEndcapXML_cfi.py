import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/dd4hep/cms.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/caloBase/2026/v1/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/dd4hep/escon.xml',
        'Geometry/HGCalCommonData/data/dd4hep/esalgo.xml',
        'Geometry/HGCalCommonData/data/fastTiming.xml',
        'Geometry/HGCalCommonData/data/fastTimingEndcap.xml',
        'Geometry/HGCalCommonData/data/v6/fastTimingElement.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)

import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v3/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/ForwardCommonData/data/fsc/2025/v1/cmsextent.xml',
        'Geometry/ForwardCommonData/data/fsc/2025/v1/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/ForwardCommonData/data/fsc/2025/v1/fsc.xml',
        'Geometry/ForwardSimData/data/fscsens.xml',
        'Geometry/ForwardSimData/data/fscProdCuts.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)

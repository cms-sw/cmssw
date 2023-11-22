import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v3/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/ForwardCommonData/data/cms.xml',
        'Geometry/ForwardCommonData/data/zdcmaterials/2021/v1/zdcmaterials.xml',
        'Geometry/ForwardCommonData/data/lumimaterials.xml',
        'Geometry/ForwardCommonData/data/zdcrotations.xml',
        'Geometry/ForwardCommonData/data/lumirotations.xml',
        'Geometry/ForwardCommonData/data/zdc/2021/v3/zdc.xml',
#       'Geometry/ForwardCommonData/data/zdclumi/2021/v2/zdclumi.xml',
        'Geometry/ForwardCommonData/data/rpd/2021/v1/rpd.xml',
        'Geometry/ForwardCommonData/data/cmszdcTest.xml',
        'Geometry/ForwardSimData/data/zdcsens/2021/v1/zdcsens.xml',
        'Geometry/ForwardSimData/data/zdcProdCuts/2021/v3/zdcProdCuts.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)

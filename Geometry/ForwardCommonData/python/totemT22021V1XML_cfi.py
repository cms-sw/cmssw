import FWCore.ParameterSet.Config as cms

## 2015 + new phase 1 pixel detector

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2021/v3/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/TrackerCommonData/data/Run2/trackermaterial.xml',
        'Geometry/ForwardCommonData/data/forward/2021/v1/forward.xml',
        'Geometry/ForwardCommonData/data/totemt2/2021/v1/totemt2.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



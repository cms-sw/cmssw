import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2023/v1/cms.xml',
        'Geometry/CMSCommonData/data/cavernData/2017/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/caloBase/2023/v1/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/ForwardCommonData/data/forwardshield/2023/v1/forwardshield.xml',
        'Geometry/ForwardCommonData/data/hfnose/v1/hfnose.xml',
        'Geometry/ForwardCommonData/data/hfnoseWafer/v1/hfnoseWafer.xml',
        'Geometry/ForwardCommonData/data/hfnoseCell/v1/hfnoseCell.xml',
        'Geometry/ForwardCommonData/data/hfnoseCons/v1/hfnoseCons.xml',
        'Geometry/ForwardCommonData/data/hfnoseDummy.xml',
        'Geometry/ForwardSimData/data/hfnosesens.xml',
        'Geometry/ForwardSimData/data/hfnoseProdCuts.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml',
        ),
    rootNodeName = cms.string('cms:OCMS')
)

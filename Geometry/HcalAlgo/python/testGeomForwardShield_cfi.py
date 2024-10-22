import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials/2021/v1/materials.xml', 
                               'Geometry/HcalCommonData/data/hcalforwardmaterial.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HcalCommonData/data/hcalrotations.xml',
                               'Geometry/CMSCommonData/data/normal/cmsextent.xml',
                               'Geometry/HcalAlgo/test/data/cms.xml',
                               'Geometry/HcalAlgo/test/data/muonBase.xml',
                               'Geometry/ForwardCommonData/data/bundle/forwardshield.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



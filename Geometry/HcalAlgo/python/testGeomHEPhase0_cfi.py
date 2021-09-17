import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
                               'Geometry/TrackerCommonData/data/trackermaterial.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HcalCommonData/data/hcalrotations.xml',
                               'Geometry/CMSCommonData/data/normal/cmsextent.xml',
                               'Geometry/HcalAlgo/test/data/cms.xml',
                               'Geometry/HcalAlgo/test/data/caloBase.xml',
                               'Geometry/HcalAlgo/test/data/muonBase.xml',
                               'Geometry/HcalCommonData/data/hcalalgo.xml',
                               'Geometry/HcalCommonData/data/hcalendcapalgo.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



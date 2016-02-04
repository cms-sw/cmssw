import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
            'Geometry/CMSCommonData/data/rotations.xml',
            'Geometry/HcalCommonData/data/hcalrotations.xml',
            'Geometry/CMSCommonData/data/normal/cmsextent.xml',
            'Geometry/HcalAlgo/test/data/cms.xml',
            'Geometry/HcalCommonData/data/hcalalgo.xml',
            'Geometry/HcalCommonData/data/hcalbarrelalgo.xml',
            'Geometry/HcalCommonData/data/hcalendcapalgo.xml',
            'Geometry/HcalCommonData/data/hcalouteralgo.xml',
            'Geometry/HcalCommonData/data/hcalforwardalgo.xml',
            'Geometry/HcalCommonData/data/hcalforwardfibre.xml',
            'Geometry/HcalCommonData/data/hcalforwardmaterial.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



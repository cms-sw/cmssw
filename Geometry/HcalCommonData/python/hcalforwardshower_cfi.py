import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
                               'Geometry/HcalCommonData/data/hcalforwardmaterial.xml', 
                               'Geometry/HcalCommonData/data/hcalforwardshower.xml',
                               'Geometry/HcalCommonData/data/Phase0/hcalSimNumbering.xml'),
    rootNodeName = cms.string('hcalforwardshower:HFWorld')
)



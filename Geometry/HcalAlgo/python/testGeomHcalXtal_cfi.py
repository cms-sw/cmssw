import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalXtal.xml'),
    rootNodeName = cms.string('TBHcal:TBHCal')
)



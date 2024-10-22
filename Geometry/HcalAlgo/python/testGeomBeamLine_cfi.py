import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06BeamLine.xml'),
    rootNodeName = cms.string('TBHcal:TBHCal')
)



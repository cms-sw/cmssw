import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalTestBeamData/test/2007/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07BeamLine.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalBarrel.xml', 
        'Geometry/HcalTestBeamData/test/2007/TBHcalEndcap.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07Sens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06SimNumbering.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06ProdCuts.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06Util.xml'),
    rootNodeName = cms.string('TBHcal:TBHCal')
)



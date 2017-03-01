import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04BeamLine.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalXtal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalCable.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalBarrel.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalEndcap.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04Sens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04SimNumbering.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04Util.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04XtalProdCuts.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04ProdCuts.xml'),
    rootNodeName = cms.string('TBHcal:TBHCal')
)



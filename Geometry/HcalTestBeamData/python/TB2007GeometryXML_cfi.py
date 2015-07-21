import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalTestBeamData/data/2007/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07BeamLine.xml', 
        'Geometry/EcalCommonData/data/eecon.xml', 
        'Geometry/EcalCommonData/data/eealgo.xml', 
        'Geometry/HcalTestBeamData/data/2007/eehier.xml', 
        'Geometry/HcalTestBeamData/data/2007/eefixed.xml', 
        'Geometry/HcalTestBeamData/data/2007/eregalgo.xml', 
        'Geometry/HcalTestBeamData/data/2007/escon.xml', 
        'Geometry/HcalTestBeamData/data/2007/esalgoTB.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalCable.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalBarrel.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalEndcap.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07Sens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06SimNumbering.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07eeSens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal07esSens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06ProdCuts.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06Util.xml'),
    rootNodeName = cms.string('TBHcal:TBHCal')
)



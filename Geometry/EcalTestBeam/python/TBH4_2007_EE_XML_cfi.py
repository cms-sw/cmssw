import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 

        'Geometry/EcalTestBeam/data/ebcon.xml', 

        'Geometry/EcalTestBeam/data/2007/eehier.xml', 
        'Geometry/EcalTestBeam/data/2007/eregalgo.xml', 
        'Geometry/EcalTestBeam/data/2007/tbrot.xml', 
        'Geometry/EcalTestBeam/data/2007/TBH4.xml', 

        'Geometry/EcalCommonData/data/eecon.xml', 
        'Geometry/EcalCommonData/data/eealgo.xml', 
        'Geometry/EcalTestBeam/data/2007/TBH4ecalsens.xml', 
        'Geometry/EcalSimData/data/EcalProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('TBH4:OCMS')
)


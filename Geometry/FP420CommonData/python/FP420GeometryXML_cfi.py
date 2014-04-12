import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/FP420CommonData/data/fp420world.xml', 
        'Geometry/FP420CommonData/data/fp420.xml', 
        'Geometry/FP420CommonData/data/zzzrectangle.xml', 
        'Geometry/FP420CommonData/data/materialsfp420.xml', 
        'Geometry/FP420CommonData/data/FP420Rot.xml', 
        'Geometry/FP420SimData/data/fp420sens.xml', 
        'Geometry/FP420SimData/data/FP420ProdCuts.xml'),
    rootNodeName = cms.string('fp420world:OCMS')
)



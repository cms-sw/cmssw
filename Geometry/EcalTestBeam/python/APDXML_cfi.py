import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/EcalTestBeam/data/ebapd.xml',
        'Geometry/EcalTestBeam/data/ebapdsens.xml'),
    rootNodeName = cms.string('ebapd:ECAL')
)

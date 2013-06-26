import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/hcalforwardmaterial.xml',
        'Geometry/HcalTestBeamData/data/TBHcal04HF.xml',
        'Geometry/HcalTestBeamData/data/TBHcal04HFBeamLine.xml',
        'Geometry/HcalTestBeamData/data/TBHcal04HFWedge.xml',
        'Geometry/HcalTestBeamData/data/TBHcal04HFSens.xml'),
    rootNodeName = cms.string('TBHcal04HF:TBHCal')
)



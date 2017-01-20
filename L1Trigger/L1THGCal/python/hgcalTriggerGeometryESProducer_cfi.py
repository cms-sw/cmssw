import FWCore.ParameterSet.Config as cms


geometry = cms.PSet( TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2'),
                     L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt"),
                     L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V7_0.txt"),
                     eeSDName = cms.string('HGCalEESensitive'),
                     fhSDName = cms.string('HGCalHESiliconSensitive'),
                     bhSDName = cms.string('HGCalHEScintillatorSensitive'),
                   )

hgcalTriggerGeometryESProducer = cms.ESProducer(
    'HGCalTriggerGeometryESProducer',
    TriggerGeometry = geometry
)

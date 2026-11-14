import FWCore.ParameterSet.Config as cms

l1tHGCalTriggerGeometryESProducer = cms.ESProducer("HGCalTriggerGeometryESProducer",
    TriggerGeometry = cms.PSet(
        DisconnectedLayers = cms.vuint32(
            2, 4, 6, 8, 10,
            12, 14, 16, 18, 20,
            22, 24, 26
        ),
        DisconnectedModules = cms.vuint32(0),
        JsonMappingFile = cms.FileInPath('L1Trigger/L1THGCal/data/hgcal_trigger_link_mapping_120links_v1.json'),
        L1TLinksMapping = cms.FileInPath('L1Trigger/L1THGCal/data/links_mapping_decentralized_signaldriven_0.txt'),
        L1TModulesMapping = cms.FileInPath('L1Trigger/L1THGCal/data/panel_mapping_V9_decentralized_jun19_0.txt'),
        ScintillatorLinksPerModule = cms.uint32(2),
        ScintillatorModuleSize = cms.uint32(6),
        ScintillatorTriggerCellSize = cms.uint32(2),
        TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp3')
    )
)

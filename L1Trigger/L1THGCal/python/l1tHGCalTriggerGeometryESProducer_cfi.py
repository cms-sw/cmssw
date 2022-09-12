import FWCore.ParameterSet.Config as cms

disconnectedTriggerLayers = [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28
        ]


geometry = cms.PSet( TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2'),
                     L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V9_decentralized_jun19_0.txt"),
                     L1TLinksMapping = cms.FileInPath('L1Trigger/L1THGCal/data/links_mapping_decentralized_signaldriven_0.txt'),
                     ScintillatorTriggerCellSize = cms.uint32(2),
                     ScintillatorModuleSize = cms.uint32(6),
                     ScintillatorLinksPerModule = cms.uint32(2),
                     DisconnectedModules = cms.vuint32(0),
                     DisconnectedLayers = cms.vuint32(disconnectedTriggerLayers)
                   )

l1tHGCalTriggerGeometryESProducer = cms.ESProducer(
    'HGCalTriggerGeometryESProducer',
    TriggerGeometry = geometry
)

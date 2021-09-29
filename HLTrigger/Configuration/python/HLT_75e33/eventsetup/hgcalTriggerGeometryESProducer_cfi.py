import FWCore.ParameterSet.Config as cms

hgcalTriggerGeometryESProducer = cms.ESProducer("HGCalTriggerGeometryESProducer",
    TriggerGeometry = cms.PSet(
        DisconnectedLayers = cms.vuint32(
            2, 4, 6, 8, 10,
            12, 14, 16, 18, 20,
            22, 24, 26, 28
        ),
        DisconnectedModules = cms.vuint32(0),
        L1TCellNeighborsBHMapping = cms.FileInPath('L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt'),
        L1TCellNeighborsMapping = cms.FileInPath('L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt'),
        L1TCellsBHMapping = cms.FileInPath('L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt'),
        L1TCellsMapping = cms.FileInPath('L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V8_0.txt'),
        L1TLinksMapping = cms.FileInPath('L1Trigger/L1THGCal/data/links_mapping_V11_decentralized_signaldriven_0.txt'),
        L1TModulesMapping = cms.FileInPath('L1Trigger/L1THGCal/data/panel_mapping_V11_decentralized_march20_1.txt'),
        ScintillatorLinksPerModule = cms.uint32(2),
        ScintillatorModuleSize = cms.uint32(6),
        ScintillatorTriggerCellSize = cms.uint32(2),
        TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
    ),
    isV9Geometry = cms.bool(True)
)

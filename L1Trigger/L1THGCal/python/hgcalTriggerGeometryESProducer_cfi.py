import FWCore.ParameterSet.Config as cms


geometry = cms.PSet( TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1'),
                     L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V8_0.txt"),
                     L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_60deg_6mod_0.txt"),
                     L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt"),
                     L1TCellsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt"),
                     L1TCellNeighborsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt"),
                     DisconnectedModules = cms.vuint32(0)
                   )

hgcalTriggerGeometryESProducer = cms.ESProducer(
    'HGCalTriggerGeometryESProducer',
    TriggerGeometry = geometry
)



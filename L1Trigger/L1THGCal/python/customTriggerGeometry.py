import FWCore.ParameterSet.Config as cms


def custom_geometry_V9(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V9_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWafersMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_mapping_V9_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_tdr_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsSciMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_sci_2x2_V9_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsSciMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_sci_2x2_V9_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerPrimitiveDigiProducer.FECodec.MaxCellsInModule = cms.uint32(288)
    process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].FECodec.MaxCellsInModule = cms.uint32(288)
    return process

def custom_geometry_ZoltanSplit_V8(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_tdr_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerPrimitiveDigiProducer.FECodec.MaxCellsInModule = cms.uint32(288)
    process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].FECodec.MaxCellsInModule = cms.uint32(288)
    return process


def custom_geometry_ZoltanSplit_V7(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_tdr_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerPrimitiveDigiProducer.FECodec.MaxCellsInModule = cms.uint32(288)
    process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].FECodec.MaxCellsInModule = cms.uint32(288)
    return process

def custom_geometry_6inch_V8(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_1.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWaferNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_neighbor_mapping_V8_0.txt")
    process.hgcalTriggerPrimitiveDigiProducer.FECodec.MaxCellsInModule = cms.uint32(116)
    process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].FECodec.MaxCellsInModule = cms.uint32(116)
    return process

def custom_geometry_6inch_V7(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_1.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWaferNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_neighbor_mapping_V7_0.txt")
    process.hgcalTriggerPrimitiveDigiProducer.FECodec.MaxCellsInModule = cms.uint32(116)
    process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].FECodec.MaxCellsInModule = cms.uint32(116)
    return process

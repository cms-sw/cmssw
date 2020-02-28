import FWCore.ParameterSet.Config as cms


def custom_geometry_decentralized_V9(process, links='signaldriven', implementation=2):
    if links=='signaldriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_decentralized_signaldriven_0.txt'
    elif links=='pudriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_decentralized_jun19_0.txt'
    else:
        raise RuntimeError('Unknown links mapping "{}". Options are "signaldriven" or "pudriven".'.format(links))
    if implementation==1:
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp1')
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V9_1.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWafersMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_mapping_V9_2.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_decentralized_jun19_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath(links_mapping)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsSciMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_sci_2x2_12x12_V9_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsSciMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_sci_2x2_12x12_V9_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorWafersPerModule = cms.uint32(1)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    elif implementation==2:
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorModuleSize = cms.uint32(6)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V9_decentralized_jun19_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath(links_mapping)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorLinksPerModule = cms.uint32(2)
    return process


def custom_geometry_V9(process, implementation=2):
    if implementation==1:
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp1')
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V9_1.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWafersMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_mapping_V9_2.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_jan19_1.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath("L1Trigger/L1THGCal/data/links_mapping_jan19_1.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsSciMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_sci_2x2_12x12_V9_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsSciMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_sci_2x2_12x12_V9_0.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorWafersPerModule = cms.uint32(3)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    elif implementation==2:
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorModuleSize = cms.uint32(12)
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V9_jan19_1.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath("L1Trigger/L1THGCal/data/links_mapping_jan19_1.txt")
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorLinksPerModule = cms.uint32(1)
    return process

def custom_geometry_ZoltanSplit_V8(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_tdr_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    return process


def custom_geometry_ZoltanSplit_V7(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_tdr_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    return process

def custom_geometry_6inch_V8(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_1.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWaferNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_neighbor_mapping_V8_0.txt")
    return process

def custom_geometry_6inch_V7(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_1.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWaferNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_neighbor_mapping_V7_0.txt")
    return process

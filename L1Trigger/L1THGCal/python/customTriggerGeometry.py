import FWCore.ParameterSet.Config as cms

def custom_geometry_ZoltanSplit_V8(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_60deg_6mod_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.eeSDName = cms.string('HGCalEESensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.fhSDName = cms.string('HGCalHESiliconSensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.bhSDName = cms.string('HGCalHEScintillatorSensitive')
    return process


def custom_geometry_ZoltanSplit_V7(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_60deg_6mod_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.eeSDName = cms.string('HGCalEESensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.fhSDName = cms.string('HGCalHESiliconSensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.bhSDName = cms.string('HGCalHEScintillatorSensitive')
    return process

def custom_geometry_6inch_V8(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_1.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWaferNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_neighbor_mapping_V8_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.eeSDName = cms.string('HGCalEESensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.fhSDName = cms.string('HGCalHESiliconSensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.bhSDName = cms.string('HGCalHEScintillatorSensitive')
    return process

def custom_geometry_6inch_V7(process):
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping_PairsRing_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_1.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TWaferNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/wafer_neighbor_mapping_V7_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.eeSDName = cms.string('HGCalEESensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.fhSDName = cms.string('HGCalHESiliconSensitive')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.bhSDName = cms.string('HGCalHEScintillatorSensitive')
    return process

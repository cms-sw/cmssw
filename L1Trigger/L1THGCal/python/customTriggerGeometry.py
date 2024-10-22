import FWCore.ParameterSet.Config as cms

def custom_geometry_V11_Imp3(process, stage1links=120):
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp3')
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
    if stage1links==120:
        process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.JsonMappingFile = cms.FileInPath("L1Trigger/L1THGCal/data/hgcal_trigger_link_mapping_120links_v1.json")
    elif stage1links==72:
        process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.JsonMappingFile = cms.FileInPath("L1Trigger/L1THGCal/data/hgcal_trigger_link_mapping_72links_v2.json")
    else:
        raise RuntimeError('{} Stage 1 input links is not supported. Supported options are 72 or 120 links'.format(stage1links))
    return process

def custom_geometry_V11_Imp2(process, links='signaldriven'):
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
    if links=='signaldriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_V11_decentralized_signaldriven_0.txt'
    elif links=='pudriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_V11_decentralized_march20_0.txt'
    else:
        raise RuntimeError('Unknown links mapping "{}". Options are "signaldriven" or "pudriven".'.format(links))
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorModuleSize = cms.uint32(6)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V11_decentralized_march20_2.txt")
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath(links_mapping)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorLinksPerModule = cms.uint32(2)
    return process


def custom_geometry_V10(process, links='signaldriven'):
    if links=='signaldriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_decentralized_signaldriven_0.txt'
    elif links=='pudriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_decentralized_jun19_0.txt'
    else:
        raise RuntimeError('Unknown links mapping "{}". Options are "signaldriven" or "pudriven".'.format(links))
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorModuleSize = cms.uint32(6)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V9_decentralized_jun19_0.txt")
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath(links_mapping)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.l1tHGCalTriggerGeometryESProducer.TriggerGeometry.ScintillatorLinksPerModule = cms.uint32(2)
    return process


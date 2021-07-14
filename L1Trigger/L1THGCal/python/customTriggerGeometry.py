import FWCore.ParameterSet.Config as cms


def custom_geometry_decentralized_V11(process, links='signaldriven',implementation=1):
    if links=='signaldriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_V11_decentralized_signaldriven_0.txt'
    elif links=='pudriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_V11_decentralized_march20_0.txt'
    else:
        raise RuntimeError('Unknown links mapping "{}". Options are "signaldriven" or "pudriven".'.format(links))
    if implementation==1:
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
    elif implementation==2:
        process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp3')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorModuleSize = cms.uint32(6)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V11_decentralized_march20_2.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath(links_mapping)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorLinksPerModule = cms.uint32(2)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.JsonMappingFile = cms.FileInPath("L1Trigger/L1THGCal/data/hgcal_trigger_link_mapping_v1.json")
    return process


def custom_geometry_decentralized_V10(process, links='signaldriven'):
    if links=='signaldriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_decentralized_signaldriven_0.txt'
    elif links=='pudriven':
        links_mapping = 'L1Trigger/L1THGCal/data/links_mapping_decentralized_jun19_0.txt'
    else:
        raise RuntimeError('Unknown links mapping "{}". Options are "signaldriven" or "pudriven".'.format(links))
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2')
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorTriggerCellSize = cms.uint32(2)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorModuleSize = cms.uint32(6)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V9_decentralized_jun19_0.txt")
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.L1TLinksMapping = cms.FileInPath(links_mapping)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedModules = cms.vuint32(0)
    process.hgcalTriggerGeometryESProducer.TriggerGeometry.ScintillatorLinksPerModule = cms.uint32(2)
    return process


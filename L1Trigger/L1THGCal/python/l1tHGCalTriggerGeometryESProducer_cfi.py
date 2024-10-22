import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16

CEE_LAYERS = 28
TOTAL_LAYERS = 50

CEE_LAYERS_V16 = 26
TOTAL_LAYERS_V16 = 47 

def disconnected_layers(ecal_layers):
    return [l for l in range(1,ecal_layers+1) if l%2==0]


geometry = cms.PSet( TriggerGeometryName = cms.string('HGCalTriggerGeometryV9Imp2'),
                     L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_V9_decentralized_jun19_0.txt"),
                     L1TLinksMapping = cms.FileInPath('L1Trigger/L1THGCal/data/links_mapping_decentralized_signaldriven_0.txt'),
                     ScintillatorTriggerCellSize = cms.uint32(2),
                     ScintillatorModuleSize = cms.uint32(6),
                     ScintillatorLinksPerModule = cms.uint32(2),
                     DisconnectedModules = cms.vuint32(0),
                     DisconnectedLayers = cms.vuint32(disconnected_layers(CEE_LAYERS))
                   )

phase2_hgcalV16.toModify(geometry, 
                         DisconnectedLayers = cms.vuint32(disconnected_layers(CEE_LAYERS_V16))
                         )

l1tHGCalTriggerGeometryESProducer = cms.ESProducer(
    'HGCalTriggerGeometryESProducer',
    TriggerGeometry = geometry
)

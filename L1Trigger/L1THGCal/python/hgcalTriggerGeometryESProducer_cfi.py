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


L1TTriggerTowerConfig_xy = cms.PSet(L1TTriggerTowerMapping=cms.FileInPath("L1Trigger/L1THGCal/data/TCmapping_v2.txt"),
                                    refCoord1=cms.double(-167.5),
                                    refCoord2=cms.double(-167.5),
                                    refZ=cms.double(320.755),
                                    binSizeCoord1=cms.double(5.),
                                    binSizeCoord2=cms.double(5.),
                                    type=cms.int32(0))

L1TTriggerTowerConfig_hgcroc_etaphi = cms.PSet(L1TTriggerTowerMapping=cms.FileInPath("L1Trigger/L1THGCal/data/TCmapping_hgcroc_eta-phi_v0.txt"),
                                               refCoord1=cms.double(1.4569),
                                               refCoord2=cms.double(-3.097959),
                                               refZ=cms.double(320.755),
                                               binSizeCoord1=cms.double(0.0938888888889),
                                               binSizeCoord2=cms.double(0.087266),
                                               type=cms.int32(1))

geometry = cms.PSet( TriggerGeometryName = cms.string('HGCalTriggerGeometryHexLayerBasedImp1'),
                     L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_8inch_aligned_192_432_V8_0.txt"),
                     L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/panel_mapping_tdr_0.txt"),
                     L1TCellNeighborsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_8inch_aligned_192_432_0.txt"),
                     L1TCellsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping_BH_3x3_30deg_0.txt"),
                     L1TCellNeighborsBHMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_neighbor_mapping_BH_3x3_30deg_0.txt"),
                     L1TTriggerTowerMapping =  cms.FileInPath("L1Trigger/L1THGCal/data/TCmapping_hgcroc_eta-phi_v0.txt"),
                     L1TTriggerTowerConfig = L1TTriggerTowerConfig_hgcroc_etaphi,
                     DisconnectedModules = cms.vuint32(0),
                     DisconnectedLayers = cms.vuint32(disconnectedTriggerLayers)
                   )

hgcalTriggerGeometryESProducer = cms.ESProducer(
    'HGCalTriggerGeometryESProducer',
    TriggerGeometry = geometry
)

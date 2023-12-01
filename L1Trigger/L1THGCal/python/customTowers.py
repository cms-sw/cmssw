import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import L1TTriggerTowerConfig_energySplit
import math

def custom_towers_unclustered_tc(process):
    process.l1tHGCalTowerProducer.InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClusteringUnclustered')
    process.l1tHGCalTowerProducerHFNose.InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer2ProducerHFNose:HGCalBackendLayer2Processor3DClusteringUnclustered')
    return process


def custom_towers_all_tc(process):
    process.l1tHGCalTowerProducer.InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering')
    process.l1tHGCalTowerProducerHFNose.InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer1ProducerHFNose:HGCalBackendLayer1Processor2DClustering')
    return process


def custom_towers_etaphi(process,
        minEta=1.479,
        maxEta=3.0,
        minPhi=-math.pi,
        maxPhi=math.pi,
        nBinsEta=18,
        nBinsPhi=72,
        binsEta=[],
        binsPhi=[]
        ):
    parameters_towers_2d = process.l1tHGCalTowerMapProducer.ProcessorParameters.towermap_parameters
    parameters_towers_2d.L1TTriggerTowerConfig.readMappingFile = cms.bool(False)
    parameters_towers_2d.L1TTriggerTowerConfig.minEta = cms.double(minEta)
    parameters_towers_2d.L1TTriggerTowerConfig.maxEta = cms.double(maxEta)
    parameters_towers_2d.L1TTriggerTowerConfig.minPhi = cms.double(minPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.maxPhi = cms.double(maxPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsEta = cms.int32(nBinsEta)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsPhi = cms.int32(nBinsPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.binsEta = cms.vdouble(binsEta)
    parameters_towers_2d.L1TTriggerTowerConfig.binsPhi = cms.vdouble(binsPhi)
    return process

def custom_towers_energySplit(process):
    parameters_towers_2d = L1TTriggerTowerConfig_energySplit.clone()
    process.hgcalTowerMapProducer.ProcessorParameters.towermap_parameters.L1TTriggerTowerConfig = parameters_towers_2d
    return process

def custom_towers_map(process,
        towermapping='L1Trigger/L1THGCal/data/tower_mapping_hgcroc_eta-phi_v3.txt',
        minEta=1.479,
        maxEta=3.0,
        minPhi=-math.pi,
        maxPhi=math.pi,
        nBinsEta=18,
        nBinsPhi=72
        ):
    parameters_towers_2d = process.l1tHGCalTowerMapProducer.ProcessorParameters.towermap_parameters
    parameters_towers_2d.L1TTriggerTowerConfig.readMappingFile = cms.bool(True)
    parameters_towers_2d.L1TTriggerTowerConfig.L1TTriggerTowerMapping = cms.FileInPath(towermapping)
    parameters_towers_2d.L1TTriggerTowerConfig.minEta = cms.double(minEta)
    parameters_towers_2d.L1TTriggerTowerConfig.maxEta = cms.double(maxEta)
    parameters_towers_2d.L1TTriggerTowerConfig.minPhi = cms.double(minPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.maxPhi = cms.double(maxPhi)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsEta = cms.int32(nBinsEta)
    parameters_towers_2d.L1TTriggerTowerConfig.nBinsPhi = cms.int32(nBinsPhi)
    return process


def custom_tower_map_hgcroc(process,
        towermapping='L1Trigger/L1THGCal/data/tower_mapping_hgcroc_eta-phi_v3.txt',
        minEta=1.479,
        maxEta=3.0,
        minPhi=-math.pi,
        maxPhi=math.pi,
        nBinsEta=18,
        nBinsPhi=72
        ):
    return custom_towers_map(process,
                             towermapping,
                             minEta,
                             maxEta,
                             minPhi,
                             maxPhi,
                             nBinsEta,
                             nBinsPhi
                             )


def custom_tower_map_wafer(process,
        towermapping='L1Trigger/L1THGCal/data/tower_mapping_wafer_eta-phi_v3.txt',
        minEta=1.479,
        maxEta=3.0,
        minPhi=-math.pi,
        maxPhi=math.pi,
        nBinsEta=18,
        nBinsPhi=72
        ):
    return custom_towers_map(process,
                             towermapping,
                             minEta,
                             maxEta,
                             minPhi,
                             maxPhi,
                             nBinsEta,
                             nBinsPhi
                             )

import FWCore.ParameterSet.Config as cms
import math

L1TTriggerTowerConfig_etaphi = cms.PSet(readMappingFile=cms.bool(False),
                                        doNose=cms.bool(False),
                                        minEta=cms.double(1.479),
                                        maxEta=cms.double(3.0),
                                        minPhi=cms.double(-1*math.pi),
                                        maxPhi=cms.double(math.pi),
                                        nBinsEta=cms.int32(18),
                                        nBinsPhi=cms.int32(72),
                                        binsEta=cms.vdouble(),
                                        binsPhi=cms.vdouble(),
                                        splitModuleSum=cms.bool(False))

L1TTriggerTowerConfig_energySplit = cms.PSet(readMappingFile=cms.bool(False),
                                        doNose=cms.bool(False),
                                        minEta=cms.double(1.305),
                                        maxEta=cms.double(3.045),
                                        minPhi=cms.double(-1*math.pi),
                                        maxPhi=cms.double(math.pi),
                                        nBinsEta=cms.int32(20),
                                        nBinsPhi=cms.int32(72),
                                        binsEta=cms.vdouble(),
                                        binsPhi=cms.vdouble(),
                                        splitModuleSum=cms.bool(True),
                                        moduleTowerMapping=cms.FileInPath("L1Trigger/L1THGCal/data/tower_per_module_silic8_scint16.txt"))

towerMap2D_parValues = cms.PSet( useLayerWeights = cms.bool(False),
                                 layerWeights = cms.vdouble(),
                                  AlgoName = cms.string('HGCalTowerMapsWrapper'),
                                 L1TTriggerTowerConfig = L1TTriggerTowerConfig_etaphi
)

tower_map = cms.PSet( ProcessorName  = cms.string('HGCalTowerMapProcessor'),
                      towermap_parameters = towerMap2D_parValues.clone()
                  )

l1tHGCalTowerMapProducer = cms.EDProducer(
    "HGCalTowerMapProducer",
    InputTriggerSums = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
    ProcessorParameters = tower_map.clone()
    )

L1TTriggerTowerConfigHFNose_etaphi = L1TTriggerTowerConfig_etaphi.clone(
    doNose = True ,
    minEta = 3.0 ,
    maxEta = 4.2
)

towerMap2DHFNose_parValues = towerMap2D_parValues.clone(
    L1TTriggerTowerConfig = L1TTriggerTowerConfigHFNose_etaphi
)

towerHFNose_map = cms.PSet( ProcessorName  = cms.string('HGCalTowerMapProcessor'),
                      towermap_parameters = towerMap2DHFNose_parValues.clone()
                  )

l1tHGCalTowerMapProducerHFNose = l1tHGCalTowerMapProducer.clone(
    InputTriggerSums = cms.InputTag('l1tHGCalConcentratorProducerHFNose:HGCalConcentratorProcessorSelection'),
    ProcessorParameters = towerHFNose_map.clone()
)

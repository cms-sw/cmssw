import FWCore.ParameterSet.Config as cms
import math

L1TTriggerTowerConfig_etaphi = cms.PSet(readMappingFile=cms.bool(False),
                                        minEta=cms.double(1.479),
                                        maxEta=cms.double(3.0),
                                        minPhi=cms.double(-1*math.pi),
                                        maxPhi=cms.double(math.pi),
                                        nBinsEta=cms.int32(18),
                                        nBinsPhi=cms.int32(72),
                                        binsEta=cms.vdouble(),
                                        binsPhi=cms.vdouble())

towerMap2D_parValues = cms.PSet( #nEtaBins = cms.int32(18),
                                 #nPhiBins = cms.int32(72),
                                 #etaBins = cms.vdouble(),
                                 #phiBins = cms.vdouble(),
                                 useLayerWeights = cms.bool(False),
                                 layerWeights = cms.vdouble(),
                                 L1TTriggerTowerConfig = L1TTriggerTowerConfig_etaphi
)

tower_map = cms.PSet( ProcessorName  = cms.string('HGCalTowerMapProcessor'),
                      towermap_parameters = towerMap2D_parValues.clone()
                  )

hgcalTowerMapProducer = cms.EDProducer(
    "HGCalTowerMapProducer",
    InputTriggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = tower_map.clone()
    )

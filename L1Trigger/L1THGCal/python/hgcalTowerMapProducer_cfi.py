import FWCore.ParameterSet.Config as cms


towerMap2D_parValues = cms.PSet( nEtaBins = cms.int32(18),
                                 nPhiBins = cms.int32(72),
                                 etaBins = cms.vdouble(),
                                 phiBins = cms.vdouble(),
                                 useLayerWeights = cms.bool(False),
                                 layerWeights = cms.vdouble()
)

tower_map = cms.PSet( ProcessorName  = cms.string('HGCalTowerMapProcessor'),
                      towermap_parameters = towerMap2D_parValues.clone()
                  )

hgcalTowerMapProducer = cms.EDProducer(
    "HGCalTowerMapProducer",
    InputTriggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = tower_map.clone()
    )

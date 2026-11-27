import FWCore.ParameterSet.Config as cms

l1tHGCalEnergySplitTowerProducer = cms.EDProducer("HGCalTowerProducer",
    InputTowerMaps = cms.InputTag("l1tHGCalEnergySplitTowerMapProducer","HGCalTowerMapProcessor"),
    InputTriggerCells = cms.InputTag("l1tHGCalBackEndLayer1Producer","HGCalBackendLayer1Processor2DClustering"),
    ProcessorParameters = cms.PSet(
        ProcessorName = cms.string('HGCalTowerProcessor'),
        includeTrigCells = cms.bool(False),
        towermap_parameters = cms.PSet(
            AlgoName = cms.string('HGCalTowerMapsWrapper'),
            L1TTriggerTowerConfig = cms.PSet(
                binsEta = cms.vdouble(),
                binsPhi = cms.vdouble(),
                doNose = cms.bool(False),
                maxEta = cms.double(3.0),
                maxPhi = cms.double(3.141592653589793),
                minEta = cms.double(1.479),
                minPhi = cms.double(-3.141592653589793),
                nBinsEta = cms.int32(18),
                nBinsPhi = cms.int32(72),
                readMappingFile = cms.bool(False),
                splitModuleSum = cms.bool(False)
            ),
            layerWeights = cms.vdouble(),
            useLayerWeights = cms.bool(False)
        )
    )
)

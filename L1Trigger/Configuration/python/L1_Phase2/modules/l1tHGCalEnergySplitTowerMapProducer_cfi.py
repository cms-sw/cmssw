import FWCore.ParameterSet.Config as cms

l1tHGCalEnergySplitTowerMapProducer = cms.EDProducer("HGCalTowerMapProducer",
    InputTriggerSums = cms.InputTag("l1tHGCalConcentratorProducer","HGCalConcentratorProcessorSelection"),
    ProcessorParameters = cms.PSet(
        ProcessorName = cms.string('HGCalTowerMapProcessor'),
        towermap_parameters = cms.PSet(
            AlgoName = cms.string('HGCalTowerMapsWrapper'),
            L1TTriggerTowerConfig = cms.PSet(
                binsEta = cms.vdouble(),
                binsPhi = cms.vdouble(),
                doNose = cms.bool(False),
                maxEta = cms.double(3.045),
                maxPhi = cms.double(3.141592653589793),
                minEta = cms.double(1.305),
                minPhi = cms.double(-3.141592653589793),
                moduleTowerMapping = cms.FileInPath('L1Trigger/L1THGCal/data/tower_per_module_silic8_scint16.txt'),
                nBinsEta = cms.int32(20),
                nBinsPhi = cms.int32(72),
                readMappingFile = cms.bool(False),
                splitModuleSum = cms.bool(True)
            ),
            layerWeights = cms.vdouble(),
            useLayerWeights = cms.bool(False)
        )
    )
)

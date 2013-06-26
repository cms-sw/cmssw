import FWCore.ParameterSet.Config as cms

# MuDepositEnergyMonitoring
muDepEnergyMonitoring = cms.EDAnalyzer("MuDepositEnergyMonitoring",
    hadS9SizeMin = cms.double(-0.5),
    OutputMEsInRootFile = cms.bool(False),
    CosmicsCollectionLabel = cms.InputTag("muons"),
    emS9SizeMin = cms.double(-0.5),
    hoS9SizeMax = cms.double(3.0),
    hoS9SizeBin = cms.int32(1000),
    hoSizeMin = cms.double(-0.5),
    hadS9SizeMax = cms.double(3.0),
    hadSizeMin = cms.double(-0.5),
    hoSizeBin = cms.int32(1000),
    emS9SizeBin = cms.int32(1000),
    hoSizeMax = cms.double(3.0),
    OutputFileName = cms.string('MuDepositEnergyMonitoring.root'),
    emSizeBin = cms.int32(1000),
    hadS9SizeBin = cms.int32(1000),
    emS9SizeMax = cms.double(3.0),
    AlgoName = cms.string('sta'),
    emSizeMin = cms.double(-0.5),
    emSizeMax = cms.double(3.0),
    hoS9SizeMin = cms.double(-0.5),
    hadSizeBin = cms.int32(1000),
    debug = cms.bool(True),
    hadSizeMax = cms.double(3.0)
)




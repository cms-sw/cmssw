import FWCore.ParameterSet.Config as cms

# MuDepositEnergyMonitoring
muDepEnergyMonitoring = cms.EDFilter("MuDepositEnergyMonitoring",
    hadS9SizeMin = cms.double(0.0),
    OutputMEsInRootFile = cms.bool(False),
    CosmicsCollectionLabel = cms.InputTag("muons"),
    emS9SizeMin = cms.double(0.0),
    hoS9SizeMax = cms.double(3.0),
    hoS9SizeBin = cms.int32(1000),
    hoSizeMin = cms.double(0.0),
    hadS9SizeMax = cms.double(3.0),
    hadSizeMin = cms.double(0.0),
    hoSizeBin = cms.int32(1000),
    emS9SizeBin = cms.int32(1000),
    hoSizeMax = cms.double(3.0),
    OutputFileName = cms.string('MuDepositEnergyMonitoring.root'),
    emSizeBin = cms.int32(1000),
    hadS9SizeBin = cms.int32(1000),
    emS9SizeMax = cms.double(3.0),
    AlgoName = cms.string('sta'),
    emSizeMin = cms.double(0.0),
    emSizeMax = cms.double(3.0),
    hoS9SizeMin = cms.double(0.0),
    hadSizeBin = cms.int32(1000),
    debug = cms.bool(True),
    hadSizeMax = cms.double(3.0)
)




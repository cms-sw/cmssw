import FWCore.ParameterSet.Config as cms

process = cms.Process("ExREG")
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'START44_V7::All'

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    calibratedPatElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
    ),
)

process.load("EgammaAnalysis.ElectronTools.calibratedPatElectrons_cfi")

# dataset to correct
process.calibratedPatElectrons.isMC = cms.bool(False)
process.calibratedPatElectrons.inputDataset = cms.string("Jan16ReReco")
process.calibratedPatElectrons.updateEnergyError = cms.bool(True)
process.calibratedPatElectrons.correctionsType = cms.int32(2)
process.calibratedPatElectrons.combinationType = cms.int32(3)
process.calibratedPatElectrons.lumiRatio = cms.double(1.0)
process.calibratedPatElectrons.verbose = cms.bool(True)
process.calibratedPatElectrons.synchronization = cms.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )


process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/cmst3/user/cmgtools/CMG/DoubleElectron/Run2011A-16Jan2012-v1/AOD/V5/PAT_CMG_V5_9_0/cmgTuple_305.root')
    fileNames = cms.untracked.vstring('/store/cernproduction/hzz4l/CMG/DoubleElectron/Run2011A-16Jan2012-v1/AOD/V5/PAT_CMG_V5_15_0/cmgTuple_999.root')
    )

process.load('EgammaAnalysis.ElectronTools.electronRegressionEnergyProducer_cfi')
process.eleRegressionEnergy.inputElectronsTag = cms.InputTag('patElectronsWithTrigger')
process.eleRegressionEnergy.rhoCollection = cms.InputTag('kt6PFJets:rho')
process.eleRegressionEnergy.energyRegressionType = cms.uint32(1)
#process.eleRegressionEnergy.inputCollectionType = cms.uint32(0)
#process.eleRegressionEnergy.useRecHitCollections = cms.bool(True)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*_*_ExREG'),
                               fileName = cms.untracked.string('electrons_tmp.root')
                                                              )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.p = cms.Path( process.eleRegressionEnergy * process.calibratedPatElectrons)
process.outpath = cms.EndPath(process.out)

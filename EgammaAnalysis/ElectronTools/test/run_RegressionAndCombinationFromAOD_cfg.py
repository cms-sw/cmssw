import FWCore.ParameterSet.Config as cms

process = cms.Process("ExREG")
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'GR_P_V42_AN3::All'

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    calibratedElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
    ),
)

process.load("EgammaAnalysis.ElectronTools.calibratedElectrons_cfi")

# dataset to correct
process.calibratedElectrons.isMC = cms.bool(False)
process.calibratedElectrons.inputDataset = cms.string("22Jan2013ReReco")
process.calibratedElectrons.updateEnergyError = cms.bool(True)
process.calibratedElectrons.correctionsType = cms.int32(2)
process.calibratedElectrons.combinationType = cms.int32(3)
process.calibratedElectrons.lumiRatio = cms.double(1.0)
process.calibratedElectrons.verbose = cms.bool(True)
process.calibratedElectrons.synchronization = cms.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )


process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring( *(
    #'/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/0CBBC6C2-42F7-E111-B1C8-0030486780B4.root'
    #'/store/relval/CMSSW_5_3_6/RelValZEE/GEN-SIM-RECO/PU_START53_V14-v1/0003/2C92DB85-E82C-E211-B8DE-003048D37560.root' 
    #'/store/data/Run2012D/DoubleElectron/AOD/PromptReco-v1/000/203/777/D82E96C2-240B-E211-AFF3-001D09F2905B.root'
    #'/store/data/Run2012A/DoubleElectron/AOD/13Jul2012-v1/00000/FEEE5F6A-26DA-E111-B08C-00266CFAE8D0.root'
    '/store/data/Run2012D/DoubleElectron/AOD/PromptReco-v1/000/205/620/9EB2C1FD-351C-E211-B0A1-003048D37666.root'
    )
        ))




process.load('EgammaAnalysis.ElectronTools.electronRegressionEnergyProducer_cfi')
process.eleRegressionEnergy.inputElectronsTag = cms.InputTag('gsfElectrons')
process.eleRegressionEnergy.inputCollectionType = cms.uint32(0)
process.eleRegressionEnergy.useRecHitCollections = cms.bool(True)
process.eleRegressionEnergy.produceValueMaps = cms.bool(True)
process.eleRegressionEnergy.regressionInputFile = cms.string("EgammaAnalysis/ElectronTools/data/eleEnergyRegWeights_WithSubClusters_VApr15.root")
process.eleRegressionEnergy.energyRegressionType = cms.uint32(2)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*_*_ExREG'),
                               fileName = cms.untracked.string('electrons.root')
                                                              )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.p = cms.Path(process.eleRegressionEnergy * process.calibratedElectrons)
process.outpath = cms.EndPath(process.out)



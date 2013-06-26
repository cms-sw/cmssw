from FWCore.ParameterSet.Config import *

process = cms.Process("runElectronID")

process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/Geometry_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

from Geometry.CaloEventSetup.CaloTopology_cfi import *

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_4_1_3/RelValZEE/GEN-SIM-RECO/START311_V2-v1/0038/5EF714E8-5A52-E011-953C-0026189438E8.root',),
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.load("RecoEgamma.ElectronIdentification.cutsInCategoriesElectronIdentificationV06_cfi")

process.electronsCiCLoose = cms.EDFilter("EleIdCutBased",
                                         src = cms.InputTag("gsfElectrons"),
                                         algorithm = cms.string("eIDCB"),
                                         threshold = cms.double(14.5),
                                         electronIDType = process.eidLooseMC.electronIDType,
                                         electronQuality = process.eidLooseMC.electronQuality,
                                         electronVersion = process.eidLooseMC.electronVersion,
                                         additionalCategories = process.eidLooseMC.additionalCategories,
                                         classbasedlooseEleIDCutsV06 = process.eidLooseMC.classbasedlooseEleIDCutsV06,
                                         etBinning = cms.bool(False),
                                         version = cms.string(""),
                                         verticesCollection = cms.InputTag('offlinePrimaryVerticesWithBS'),
                                         reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                         reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                         )

process.eIDSequence = cms.Sequence(process.eidLooseMC)
process.CiCLooseFilter = cms.Sequence(process.electronsCiCLoose)
process.p = cms.Path(process.eIDSequence*process.CiCLooseFilter)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *', 
                                                                      'keep *_electronsCiCLoose_*_*'),
                               #SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filter')),
                               fileName = cms.untracked.string('electrons.root')
                               )

process.outpath = cms.EndPath(process.out)


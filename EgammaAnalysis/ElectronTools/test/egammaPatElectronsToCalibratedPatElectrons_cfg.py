import FWCore.ParameterSet.Config as cms
import os

from TrackingTools.Configuration.TrackingTools_cff import *

process = cms.Process("NewEScale")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    calibratedPatElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(1),
        engineName = cms.untracked.string('TRandom3')
    ),
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring
    (
    #'file:/data_CMS/cms/charlot/Run2011/AllCandidatesEPS11/HZZCandidates.root'
    'file:/afs/cern.ch/user/m/mdalchen/private/Reco/ScaleCorr/CMSSW_5_3_4/src/EgammaAnalysis/ElectronTools/test/electrons.root'
    #'/store/data/Summer11/DYToEE_M-800_TuneZ2_7TeV-pythia6-tauola/AODSIM/PU_S3_START42_V11-v2/0000/0ABF7CD0-8888-E011-8561-1CC1DE051038.root'    
    ),
#    eventsToProcess = cms.untracked.VEventRange('173243:16706390')   
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
#    fileName = cms.untracked.string('CandidateZ_newEscale.root')
    fileName = cms.untracked.string('testMC.root')
)

process.load("EgammaAnalysis.ElectronTools.calibratedPatElectrons_cfi")

# dataset to correct
#process.calibratedGsfElectrons.inputDataset = cms.string("Jan16ReReco")
#process.calibratedGsfElectrons.inputDataset = cms.string("ReReco")
process.calibratedPatElectrons.inputDataset = cms.string("Summer12_DR53X_HCP2012")
#process.calibratedGsfElectrons.inputDataset = cms.string("Fall11")
process.calibratedPatElectrons.isMC = cms.bool(True)
process.calibratedPatElectrons.isAOD = cms.bool(True)
process.calibratedPatElectrons.updateEnergyError = cms.bool(True)
process.calibratedPatElectrons.applyCorrections = cms.int32(1)
process.calibratedPatElectrons.verbose = cms.bool(True)
process.calibratedPatElectrons.synchronization = cms.bool(False)


process.p = cms.Path(process.calibratedPatElectrons)

process.outpath = cms.EndPath(process.out)
#process.GlobalTag.globaltag = 'GR_R_42_V18::All'
process.GlobalTag.globaltag = 'START53_V10::All'





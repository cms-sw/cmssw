# produce pat::Tau collection with the new DNN Tau-Ids from miniAOD 12Apr2018_94X_mc2017

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.parseArguments()

process = cms.Process('runDeepTauIDsOnMiniAOD')
process.options = cms.untracked.PSet()
process.options.wantSummary = cms.untracked.bool(False)
process.options.allowUnscheduled = cms.untracked.bool(True)
process.options.numberOfThreads = cms.untracked.uint32(8)
process.options.numberOfStreams = cms.untracked.uint32(0)

process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.GlobalTag.globaltag = '94X_mc2017_realistic_v14'
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(
    # File from dataset /GluGluHToTauTau_M125_13TeV_powheg_pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM
    '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/0498CD6A-CC42-E811-95D3-008CFA1CB8A8.root'
))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

import RecoTauTag.RecoTau.runTauIdMVA as tauIdConfig
tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, cms, debug = True,
                    toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "deepTau2017v1", "DPFTau_2016_v0",
                               "DPFTau_2016_v1"])
tauIdEmbedder.runTauID()

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple_newTauIDs.root'),
                               outputCommands = cms.untracked.vstring('drop *', "keep *_NewTauIDsEmbedded_*_*"))

process.p = cms.Path(
    process.rerunMvaIsolationSequence *
    process.NewTauIDsEmbedded
)

process.outpath = cms.EndPath(process.out)

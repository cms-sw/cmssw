import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.StandardSequences.Eras import eras
from Configuration.AlCa.GlobalTag import GlobalTag

import subprocess
import sys

options = VarParsing.VarParsing()

options.register('globalTag',
                 '124X_dataRun3_Prompt_v4', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Global Tag")

options.register('nEvents',
                 1000, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Maximum number of processed events")

options.register('inputFile',
                 '/store/data/Run2022E/ZeroBias/RAW/v1/000/359/751/00000/6ee95dd0-8fb4-4693-90b9-f7a3fbd2fdeb.root', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "EOS folder with input files")

options.register('ntupleName',
                 './MuDPGNtuple_nanoAOD_ZeroBias.root', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Folder and name ame for output ntuple")

options.parseArguments()

process = cms.Process("MUNTUPLES",eras.Run3)

process.load('FWCore.MessageService.MessageLogger_cfi')

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True),
                                        numberOfThreads = cms.untracked.uint32(4))
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.nEvents))

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string(options.globalTag)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFile),
                            secondaryFileNames = cms.untracked.vstring()
)

process.load('Configuration/StandardSequences/GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('RecoLocalMuon.Configuration.RecoLocalMuon_cff')

process.load('DPGAnalysis.MuonTools.muNtupleProducerRaw_cff')

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
process.muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()

process.nanoMuDPGPath = cms.Path(process.muonDTDigis
                                 + process.muonCSCDigis
                                 + process.muonRPCDigis
                                 + process.muonGEMDigis
                                 + process.rpcRecHits
                                 + process.gemRecHits
                                 + process.muNtupleProducerRaw)

process.load("PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff")

process.out = cms.OutputModule("NanoAODOutputModule",
                               fileName = cms.untracked.string(options.ntupleName),
                               outputCommands = process.NANOAODEventContent.outputCommands \
                                                + ["keep nanoaodFlatTable_*_*_*",
                                                   "drop edmTriggerResults_*_*_*"],
                               SelectEvents = cms.untracked.PSet(
                                              SelectEvents=cms.vstring("nanoMuDPGPath")
                               )
)

process.end = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.nanoMuDPGPath, process.end)

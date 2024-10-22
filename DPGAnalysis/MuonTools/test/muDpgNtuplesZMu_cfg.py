import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.StandardSequences.Eras import eras
from Configuration.AlCa.GlobalTag import GlobalTag

import subprocess
import sys

options = VarParsing.VarParsing()

options.register('globalTag',
                 '125X_dataRun3_relval_v4', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Global Tag")

options.register('nEvents',
                 1000, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Maximum number of processed events")

options.register('inputFile',
                 '/store/relval/CMSSW_12_6_0_pre5/SingleMuon/FEVTDEBUGHLT/125X_dataRun3_HLT_relval_v3_RelVal_2022C-v2/2590000//053845fa-aa05-48a3-8bc0-c833cfdd3e53.root', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "EOS folder with input files")

options.register('ntupleName',
                 './MuDPGNtuple_nanoAOD_ZMuSkim.root', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Folder and name ame for output ntuple")

options.register('runOnMC',
                 False, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Apply customizations to run on MC")

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

process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')

process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('DPGAnalysis.MuonTools.muNtupleProducer_cff')

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
process.muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()

process.nanoMuDPGPath = cms.Path(process.muonDTDigis
                                 + process.muonRPCDigis
                                 + process.muonGEMDigis
                                 + process.twinMuxStage2Digis
                                 + process.bmtfDigis
                                 + process.muNtupleProducer)

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

if options.runOnMC :
    from DPGAnalysis.MuonTools.customiseMuNtuples_cff import customiseForRunningOnMC
    customiseForRunningOnMC(process,"p")

import FWCore.ParameterSet.Config as cms
import subprocess

process = cms.Process("PrimitiveDigisUnpacker")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2016_cff')
process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.globaltag = "92X_dataRun2_Express_v7"
#process.GlobalTag.globaltag = "101X_dataRun2_Express_v7"


process.load("EventFilter.RPCRawToDigi.RPCCPPFRawToDigi_sqlite_cff")

process.load("EventFilter.RPCRawToDigi.rpcPacker_cfi")
process.rpcpacker.InputLabel = cms.InputTag("rpcCPPFRawToDigi")

process.load("EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi")
process.rpcUnpackingModulePacked = process.rpcUnpackingModule.clone()
process.rpcUnpackingModulePacked.InputLabel = cms.InputTag("rpcpacker")

process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")
from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *
process.rpcunpacker.InputLabel = 'rawDataCollector'

process.load('RecoLocalMuon.RPCRecHit.rpcRecHits_cfi')
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *
process.rpcRecHits.rpcDigiLabel = 'rpcunpacker'

process.load('L1Trigger.RPCTriggerPrimitives.primitiveRPCProducer_cfi')
from L1Trigger.RPCTriggerPrimitives.primitiveRPCProducer_cfi import *
process.primitiveRPCProducer.Primitiverechitlabel = 'rpcunpacker'

process.load('L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi')
from L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi import *
#process.emulatorCppfDigis.recHitLabel = 'rpcRecHits'
process.emulatorCppfDigis.recHitLabel = "primitiveRPCProducer" # For new digis

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Input source
eos_cmd = '/afs/cern.ch/project/eos/installation/ams/bin/eos.select'
readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",
        fileNames = readFiles,
)
in_dir_name = '/eos/cms/store/data/Run2018D/SingleMuon/RAW-RECO/ZMu-PromptReco-v2/000/321/457/00000/'

#readFiles.extend( cms.untracked.vstring('file:'+in_dir_name+'EC0F3940-C3A5-E811-9736-02163E01A00E.root') )

iFile = 0
for in_file_name in subprocess.check_output([eos_cmd, 'ls', in_dir_name]).splitlines():
    if not ('.root' in in_file_name): continue
    iFile += 1
    readFiles.extend( cms.untracked.vstring('file:'+in_dir_name+in_file_name) )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
#process.maxLuminosityBlocks = cms.untracked.PSet(input = cms.untracked.int32(10))

process.p = cms.Path(   process.rpcUnpackingModule *
			process.rpcCPPFRawToDigi * 
		        process.rpcpacker 
                         + 
                        process.rpcUnpackingModulePacked + 
			process.rpcunpacker * 
			process.rpcRecHits * 
			process.primitiveRPCProducer +
			process.emulatorCppfDigis 
)

# Output
process.out = cms.OutputModule("PoolOutputModule"
                               , outputCommands = cms.untracked.vstring("drop *",
									"keep *_rpcunpacker_*_*",
                                                                        "keep *_rpcRecHits_*_*",
									"keep *_emulatorCppfDigis_*_*",
							                "keep *_primitiveRPCProducer_*_*",
                                                                        "keep *_rpcCPPFRawToDigi_*_*")
                               , fileName = cms.untracked.string("TriggerPrimitive_RAW.root")
                               , SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
)

process.e = cms.EndPath(process.out)


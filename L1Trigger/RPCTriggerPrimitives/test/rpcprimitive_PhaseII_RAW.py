import FWCore.ParameterSet.Config as cms
import subprocess

process = cms.Process("PrimitiveDigisUnpacker")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D38Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D38_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load('RecoLocalMuon.RPCRecHit.rpcRecHits_cfi')
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *
process.rpcRecHits.rpcDigiLabel = 'simMuonRPCDigis'

process.load('L1Trigger.RPCTriggerPrimitives.primitiveRPCProducer_cfi')
from L1Trigger.RPCTriggerPrimitives.primitiveRPCProducer_cfi import *
process.primitiveRPCProducer.Primitiverechitlabel = 'simMuonRPCDigis'
process.primitiveRPCProducer.ClusterSizeCut = 4
process.primitiveRPCProducer.ApplyLinkBoardCut = False

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

in_dir_name='/eos/cms/store/group/dpg_dt/comm_dt/TriggerSimulation/SamplesReco/SingleMu_FlatPt-2to100/Version_10_5_0/'

readFiles.extend( cms.untracked.vstring('file:'+in_dir_name+'SimRECO_1.root') )

#iFile = 0
#for in_file_name in subprocess.check_output([eos_cmd, 'ls', in_dir_name]).splitlines():
#    if not ('.root' in in_file_name): continue
#    iFile += 1
#    readFiles.extend( cms.untracked.vstring('file:'+in_dir_name+in_file_name) )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
#process.maxLuminosityBlocks = cms.untracked.PSet(input = cms.untracked.int32(10))

process.p = cms.Path(   
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


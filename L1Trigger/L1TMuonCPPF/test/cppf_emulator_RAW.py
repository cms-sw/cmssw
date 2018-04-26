
import FWCore.ParameterSet.Config as cms
import datetime
import random

process = cms.Process('DIGI')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load('Configuration.Geometry.GeometryDB_cff')
#process.load('Configuration.StandardSequences.GeometryExtended_cff')
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

process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")
import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()
muonRPCDigis.InputLabel = 'rawDataCollector'

process.load('RecoLocalMuon.RPCRecHit.rpcRecHits_cfi')
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *

process.load('L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi')
from L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi import *

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


# Input source
process.source = cms.Source(
    'PoolSource',fileNames = cms.untracked.vstring ('/store/data/Run2017F/SingleMuon/RAW/v1/000/306/125/00000/4EDD5765-B3C0-E711-B906-02163E01A2D5.root')
    )

process.options = cms.untracked.PSet(
    )

process.treeOut = cms.OutputModule("PoolOutputModule",
                                   fileName = cms.untracked.string('test_cppf_emulator.root'),
                                   outputCommands = cms.untracked.vstring('drop *',
                                                                          "keep *_rpcunpacker_*_*", 
                                                                          "keep *_emulatorCppfDigis_*_*",
                                                                          "keep *_rpcRecHits_*_*"
                                                                          #"keep *"
                                                                          )
                                   )


# Additional output definition

# Other statements

# process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


process.rpcunpacker.InputLabel = 'rawDataCollector'
process.rpcRecHits.rpcDigiLabel = 'rpcunpacker'
process.emulatorCppfDigis.recHitLabel = 'rpcRecHits'

# Path and EndPath definitions
process.path_step =  cms.Path(process.rpcunpacker)
process.rpcrechits_step = cms.Path(process.rpcRecHits)
process.emulatorCppfDigis_step = cms.Path(process.emulatorCppfDigis)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.treeOut_step = cms.EndPath(process.treeOut)



# Schedule definition
process.schedule = cms.Schedule(
    process.path_step,
    process.rpcrechits_step,
    process.emulatorCppfDigis_step,
    process.endjob_step,
    process.treeOut_step )

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Customisation from command line
# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt10_pythia8_cfi.py -s GEN,SIM,DIGI --pileup=NoPileUp --geometry DB --conditions=auto:run1_mc --eventcontent FEVTDEBUGHLT --no_exec -n 30
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

process.load('RecoLocalMuon.RPCRecHit.rpcRecHits_cfi')
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *

process.load('L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi')
from L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi import *

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1)
process.MessageLogger = cms.Service("MessageLogger")
process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(300)
	)


# Input source
process.source = cms.Source("EmptySource"
			    )
process.options = cms.untracked.PSet(
	)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
	annotation = cms.untracked.string('SingleMuPt10_pythia8_cfi.py nevts:100'),
	name = cms.untracked.string('Applications'),
	version = cms.untracked.string('$Revision: 1.19 $')
	)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
					      SelectEvents = cms.untracked.PSet(
		SelectEvents = cms.vstring('generation_step')
		),
					      dataset = cms.untracked.PSet(
		dataTier = cms.untracked.string(''),
		filterName = cms.untracked.string('')
		),
					      eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
					      fileName = cms.untracked.string('SingleMuPt10_pythia8_cfi_py_GEN_SIM_DIGI.root'),
					      outputCommands = cms.untracked.vstring('drop *',"keep *_emulatorMuonRPCDigis_*_*", "keep *_emulatorCppfDigis_*_*", 
										     "keep *_rpcRecHits_*_*", "keep *_genParticles_*_*"),
					      #outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
					      splitLevel = cms.untracked.int32(0)
					      )

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randHelper.populate()

process.RandomNumberGeneratorService.saveFileName =  cms.untracked.string("RandomEngineState.log")


process.generator = cms.EDFilter("Pythia8PtGun",
				 PGunParameters = cms.PSet(
		AddAntiParticle = cms.bool(True),
		MaxEta = cms.double(1.6),
		MaxPhi = cms.double(3.14159265359),
		MaxPt = cms.double(30.1),
		MinEta = cms.double(1.2),
		MinPhi = cms.double(-3.14159265359),
		MinPt = cms.double(1.1),
		ParticleID = cms.vint32(-13)
		),
				 PythiaParameters = cms.PSet(
		parameterSets = cms.vstring()
		),
				 Verbosity = cms.untracked.int32(0),
				 firstRun = cms.untracked.uint32(1),
				 psethack = cms.string('single mu pt 10')
				 )

process.rpcRecHits.rpcDigiLabel = 'simMuonRPCDigis'
process.emulatorCppfDigis.recHitLabel = 'rpcRecHits'

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.rpcrechits_step = cms.Path(process.rpcRecHits)
process.emulatorCppfDigis_step = cms.Path(process.emulatorCppfDigis)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)


# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.rpcrechits_step,process.emulatorCppfDigis_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 
	
	
	# Customisation from command line
	# Add early deletion of temporary data products to reduce peak memory need
	from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
	process = customiseEarlyDelete(process)
# End adding early deletion

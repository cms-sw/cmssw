import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include DQMStore, needed by the famosSimHits
process.DQMStore = cms.Service( "DQMStore")

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate ttbar events
process.load("Configuration.Generator.TTbar_cfi")


# Common inputs, with fake conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.load('FastSimulation.Configuration.Geometries_cff')

# vertex smearing
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')

# L1 Emulator and HLT Setup
#process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_GRun_cff")

# Simulation sequence
#process.simulation = cms.Sequence(process.ProductionFilterSequence*process.simulationWithFamos)
process.source = cms.Source("EmptySource")
process.simulation = cms.Sequence( process.dummyModule )

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator)
process.simulation_step = cms.Path(process.VtxSmeared*process.simulationWithFamos)
process.reconstruction_step = cms.Path(process.reconstructionWithFamos)

# Only events accepted by L1 + HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.simulation*process.reconstructionWithFamos)
# In alternative, to reconstruct all events:
#process.HLTEndSequence = cms.Sequence()

# Schedule the HLT paths (and allows HLTAnalyzers for this test):
from FastSimulation.HighLevelTrigger.HLTSetup_cff import hltL1GtTrigReport
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.HLTAnalyzerEndpath = cms.EndPath( hltL1GtTrigReport + process.hltTrigReport )
process.HLTSchedule.append(process.HLTAnalyzerEndpath)
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.reconstruction_step)
process.schedule.extend(process.HLTSchedule)



# If uncommented : All events are reconstructed, including those rejected at L1/HLT
#process.reconstruction = cms.Path(process.reconstructionWithFamos)
#process.schedule.append(process.reconstruction)


# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
process.load('SimGeneral.MixingModule.mix_2012_Startup_50ns_PoissonOOTPU_cfi')
from FastSimulation.Configuration.MixingModule_Full2Fast import prepareGenMixing
process = prepareGenMixing(process)


# Get frontier conditions   - not applied in the HCAL, see below
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(process.GlobalTag,'auto:run2_mc_GRun')

# Apply ECAL miscalibration 
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *

# Apply Tracker misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.misalignedDTGeometry.applyAlignment = True
process.misalignedCSCGeometry.applyAlignment = True


# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTestWithHLT.root')
)
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.outpath)

# Keep the logging output to a nice level #
# process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.categories.append('HLTrigReport')
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

# PostLS1
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
process = customisePostLS1(process)

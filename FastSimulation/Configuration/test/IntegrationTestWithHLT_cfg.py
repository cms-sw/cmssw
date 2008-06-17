import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Generate ttbar events
process.load("FastSimulation.Configuration.ttbar_cfi")

# L1 Menu and prescale factors : useful for testing all L1 paths
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_cff")

# All event accepted by HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# No reconstruction - only HLT
# process.HLTEndSequence = cms.Sequence(process.dummyModule)

# All events are reconstructed, including those rejected at L1/HLT
# process.HLTEndSequence = cms.Sequence(process.dummyModule)
# reconstruction = cms.Path( reconstructionWithFamos )

# Simulation sequence
process.simulation = cms.Sequence(process.simulationWithFamos)
# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

# Get frontier conditions
# Values for globaltag are "STARTUP_V1::All", "1PB::All", "10PB::All", "IDEAL_V2::All"
process.GlobalTag.globaltag = "IDEAL_V2::All"

# Apply ECAL miscalibration
process.caloRecHits.RecHitsFactory.doMiscalib = True

# Apply Tracker misalignment (ideal alignment though)
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True

# Apply HCAL miscalibration (not ideal in that case)
process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0

# Note : if your process is not called HLT, you have to change that! 
# process.hltTrigReport.HLTriggerResults = TriggerResults::PROD
# process.hltHighLevel.TriggerResultsTag = TriggerResults::PROD 

# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTestWithHLT.root')
)
process.outpath = cms.EndPath(process.o1)

# Keep the logging output to a nice level #
# process.Timing =  cms.Service("Timing")
# process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt")



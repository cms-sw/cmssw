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

# Famos sequences (with frontier conditions)
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# L1 Menu and prescale factors : useful for testing all L1 paths
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# Reconstruction of all events, including those rejected at L1/HLT
# Uncomment this line (and the path reconstruction)
# sequence HLTEndSequence = { dummyModule }
# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT.csh
process.load("FastSimulation.Configuration.test.HLT_cff")

# All event accepted by HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)
# No reconstruction - only HLT
# process.HLTEndSequence = cms.Sequence(process.dummyModule)

# Simulation sequence
process.simulation = cms.Sequence(process.simulationWithFamos)
# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

# Note : if your process is not called HLT, you have to change that! 
#replace hltTrigReport.HLTriggerResults = TriggerResults::PROD
#replace hltHighLevel.TriggerResultsTag = TriggerResults::PROD 
# Reconstruction of all events, including those rejected at L1/HLT
# Uncomment this path
# path reconstruction = { reconstructionWithFamos } 


# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTestWithHLT.root')
)
process.outpath = cms.EndPath(process.o1)

process.GlobalTag.globaltag = cms.InputTag("IDEAL_V2","","All")
process.caloRecHits.RecHitsFactory.doMiscalib = True
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.00



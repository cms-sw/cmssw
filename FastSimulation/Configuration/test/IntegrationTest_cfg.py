import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Generate ttbar events
process.load("FastSimulation.Configuration.ttbar_cfi")

# Famos sequences (NO HLT)
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

# If you want to turn on/off pile-up
process.famosPileUp.PileUpSimulator.averageNumber = 5.0
# Parametrized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

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

# Famos with everything !
process.p1 = cms.Path(process.famosWithEverything)

# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTest.root')
)
process.outpath = cms.EndPath(process.o1)

# Keep output to a nice level
# process.Timing =  cms.Service("Timing")
# process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt")

# The following comments couldn't be translated into the new config version:

# Apply ECAL miscalibration (ideal calibration though) and HCAL miscalibration

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Generate ttbar events
process.load("FastSimulation.Configuration.ttbar_cfi")

# Famos sequences (NO HLT)
process.load("FastSimulation.Configuration.CommonInputs_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTest.root')
)

process.p1 = cms.Path(process.famosWithEverything)
process.outpath = cms.EndPath(process.o1)
process.famosPileUp.PileUpSimulator.averageNumber = 5.0
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.GlobalTag.globaltag = cms.InputTag("IDEAL_V1","","All")
process.caloRecHits.RecHitsFactory.doMiscalib = True
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0


import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30)
)

# Include DQMStore, needed by the famosSimHits
process.DQMStore = cms.Service( "DQMStore")

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate ttbar events
process.load("Configuration.Generator.TTbar_cfi")

# Famos sequences (NO HLT)
#process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.load('FastSimulation.Configuration.Geometries_cff')

# vertex smearing
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
# If you want to turn on/off pile-up
process.load('SimGeneral.MixingModule.mix_2012_Startup_50ns_PoissonOOTPU_cfi')
from FastSimulation.Configuration.MixingModule_Full2Fast import prepareGenMixing
process = prepareGenMixing(process)

# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

# Get frontier conditions    - not applied in the HCAL, see below
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(process.GlobalTag,'auto:run1_mc')
# Allow reading of the tracker geometry from the DB
process.load('CalibTracker/Configuration/Tracker_DependentRecords_forGlobalTag_nofakes_cff')

# Apply ECAL miscalibration
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *

# Apply Tracker misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.misalignedDTGeometry.applyAlignment = True
process.misalignedCSCGeometry.applyAlignment = True

#  Attention ! for the HCAL IDEAL==STARTUP
#process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
#process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
#process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_0.0.xml"

# Famos with everything !
#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything)
process.source = cms.Source("EmptySource")
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.famosWithEverything)

# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTest.root')
)
process.outpath = cms.EndPath(process.o1)

# Keep output to a nice level
# process.Timing =  cms.Service("Timing")
# process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
# process.MessageLogger.categories.append("FamosManager")
# process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                 default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                 FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))


# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

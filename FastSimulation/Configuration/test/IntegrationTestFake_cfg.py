import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include DQMStore, needed by the famosSimHits
process.DQMStore = cms.Service( "DQMStore")

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=180GeV/c2
process.load("Configuration.Generator.H200ZZ4L_cfi")
# Generate ttbar events
#  process.load("FastSimulation/Configuration/ttbar_cfi")
# Generate multijet events with different ptHAT bins
#  process.load("FastSimulation/Configuration/QCDpt80-120_cfi")
#  process.load("FastSimulation/Configuration/QCDpt600-800_cfi")
# Generate Minimum Bias Events
#  process.load("FastSimulation/Configuration/MinBiasEvents_cfi")
# Generate muons with a flat pT particle gun, and with pT=10.
# process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
# replace FlatRandomPtGunSource.PGunParameters.PartID={130}
# Generate di-electrons with pT=35 GeV
# process.load("FastSimulation/Configuration/DiElectrons_cfi")

# Famos sequences (MC conditions, not Fake anymore!)
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.load('FastSimulation.Configuration.Geometries_cff')
#process.load("FastSimulation.Configuration.CommonInputs_cff")
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag = autoCond['mc']
# Allow reading of the tracker geometry from the DB
#process.load('CalibTracker/Configuration/Tracker_DependentRecords_forGlobalTag_nofakes_cff')

# vertex smearing
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
process.load('SimGeneral.MixingModule.mix_2012_Startup_50ns_PoissonOOTPU_cfi')
from FastSimulation.Configuration.MixingModule_Full2Fast import prepareGenMixing
process = prepareGenMixing(process)

# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

#process.load('FastSimulation.Configuration.HLT_GRun_cff')

# Famos with everything !
#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything)
process.source = cms.Source("EmptySource")
process.simulation = cms.Path(process.generator*process.VtxSmeared*process.famosWithEverything)

# To write out events (not need: FastSimulation _is_ fast!)
#process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('keep *', 
                                                                     'drop *_mix_*_*'),
                              #                              process.AODSIMEventContent,
                              fileName = cms.untracked.string('MyFirstFamosFile_2.root')
                              )
process.outpath = cms.EndPath(process.o1)

# Keep output to a nice level
# process.Timing =  cms.Service("Timing")
# process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo.txt")

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

process.schedule = cms.Schedule( process.simulation, process.outpath )

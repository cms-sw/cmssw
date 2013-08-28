import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Generate ttbar events
process.load("Configuration.Generator.TTbar_cfi")
#process.load("Configuration.Generator.SingleMuPt100_cfi")

# --- This was for 2e30 :
# process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# --- This is for 8e29 :NEW DEFAULT 
#process.load('L1Trigger/Configuration/L1StartupConfig_cff')
#process.load('L1TriggerConfig/L1GtConfigProducers/Luminosity/startup/L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')

# --- This is for 1e31 :
#process.load('L1TriggerConfig/L1GtConfigProducers/Luminosity/lumi1031/L1Menu_MC2009_v0_L1T_Scales_20080922_Imp0_Unprescaled_cff')

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetupWithL1ParamMuon_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_8E29_cff")

# Only event accepted by L1 + HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# Schedule the HLT paths
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)

# If uncommented : All events are reconstructed, including those rejected at L1/HLT
process.reconstruction = cms.Path(process.reconstructionWithFamos)
process.schedule.append(process.reconstruction)

# Simulation sequence (without muon digitization)
#process.simulation = cms.Sequence(process.ProductionFilterSequence*process.simulationWithFamos)
process.source = cms.Source("EmptySource")
process.simulation = cms.Sequence(process.generator*process.simulationNoMuonDigiWithFamos)

# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

# Get frontier conditions   - not applied in the HCAL, see below
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

# Apply ECAL and HCAL miscalibration 
process.ecalRecHit.doMiscalib = True
process.hbhereco.doMiscalib = True
process.horeco.doMiscalib = True
process.hfreco.doMiscalib = True

# Apply Tracker misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.misalignedDTGeometry.applyAlignment = True
process.misalignedCSCGeometry.applyAlignment = True

# Attention ! for the HCAL IDEAL==STARTUP
# process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_0.0.xml"


# Note : if your process is not called HLT, you have to change that! 
# process.hltTrigReport.HLTriggerResults = TriggerResults::PROD
# process.hltHighLevel.TriggerResultsTag = TriggerResults::PROD 

# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTestWithHLTandL1ParamMuon.root')
)
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.outpath)

# Keep the logging output to a nice level #
# process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.categories.append('HLTrigReport')
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfoWithL1ParamMuon.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )


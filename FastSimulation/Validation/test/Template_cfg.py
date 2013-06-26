import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate ttbar events
process.load("FastSimulation.Configuration.ttbar_cfi")

# L1 Menu and prescale factors : useful for testing all L1 paths
# Note: the L1 conditions and menu now come from the GlobalTag !!!!!!!!!!!!
###process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# Other choices are
# L1 Menu 2008 2x10E30 - Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1030/L1Menu2008_2E30.cff")
# L1 Menu 2008 2x10E30 - No Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1030/L1Menu2008_2E30_Unprescaled.cff")
# L1 Menu 2008 2x10E31 - Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1031/L1Menu2008_2E31.cff")
# L1 Menu 2008 2x10E31 - No Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1031/L1Menu2008_2E31_Unprescaled.cff")
# L1 Menu 2008 10E32 - Prescale 
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumix1032/L1Menu2007.cff")
# L1 Menu 2008 10E32 - No Prescale 
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumix1032/L1Menu2007_Unprescaled.cff")

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_cff")

# Only event accepted by L1 + HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# Schedule the HLT paths
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)

# If uncommented : All events are reconstructed, including those rejected at L1/HLT
process.reconstruction = cms.Path(process.reconstructionWithFamos)
process.schedule.append(process.reconstruction)

# Simulation sequence
process.simulation = cms.Sequence(process.simulationWithFamos)
# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

# Get frontier conditions   - not applied in the HCAL, see below
process.GlobalTag.globaltag = "IDEAL_31X::All"


# Set the early collions 10TeV parameters (as in the standard RelVals)
process.famosSimHits.VertexGenerator.SigmaZ=cms.double(3.8)
process.famosSimHits.VertexGenerator.Emittance = cms.double(7.03e-08)
process.famosSimHits.VertexGenerator.BetaStar = cms.double(300.0)

# Apply ECAL and HCAL miscalibration 
process.caloRecHits.RecHitsFactory.doMiscalib = False

# Apply Tracker misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True

# Apply HCAL miscalibration (not ideal in that case).
# Choose between hcalmiscalib_startup.xml , hcalmiscalib_1pb.xml , hcalmiscalib_10pb.xml (startup is the default)
process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
#process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_startup.xml"


# Note : if your process is not called HLT, you have to change that! 
# process.hltTrigReport.HLTriggerResults = TriggerResults::PROD
# process.hltHighLevel.TriggerResultsTag = TriggerResults::PROD 

# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTDEBUGHLTEventContent,
    fileName = cms.untracked.string('Output.root'),
    dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('GEN-SIM-DIGI-HLTDEBUG-RECO')
    ) 
)
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.outpath)

process.configurationMetadata = cms.untracked.PSet(
       version = cms.untracked.string('$Revision: 1.7 $'),
          name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/FastSimulation/Validation/test/Template_cfg.py,v $'),
          annotation = cms.untracked.string('RelVal Fast Sim ==SAMPLE== IDEAL')
       ) 

# Keep the logging output to a nice level #
# process.Timing =  cms.Service("Timing")
# process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt")



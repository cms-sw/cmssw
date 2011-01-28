import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

#process.load("RecoParticleFlow.Configuration.Sources/source_singleTau_cfi")
#process.load("FastSimulation.Configuration.DiJets_cfi")
process.load("CommonTools.ParticleFlow.Sources/source_ZtoMus_cfi")


process.source.maxEventsToPrint = cms.untracked.int32(2)
process.source.pythiaPylistVerbosity = cms.untracked.int32(1)


# L1 Menu and prescale factors : useful for testing all L1 paths
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
# process.load("Configuration.StandardSequences.MagneticField_38T_cff")
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
# process.reconstruction = cms.Path(process.reconstructionWithFamos)
# process.schedule.append(process.reconstruction)

# Simulation sequence
process.simulation = cms.Sequence(process.simulationWithFamos)
# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.GlobalTag.globaltag = "IDEAL_V9::All"


# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.outpath)





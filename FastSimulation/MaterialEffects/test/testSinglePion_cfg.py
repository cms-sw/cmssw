import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Histograms
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source(
    "FlatRandomEGunSource",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.untracked.vint32(211),
        MinEta = cms.untracked.double(-3.0),
        MaxEta = cms.untracked.double(3.0),
        MinPhi = cms.untracked.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.untracked.double(3.14159265359),
        MinE = cms.untracked.double(15.0),
        MaxE = cms.untracked.double(15.0)
    ),
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)   
)

# Famos SimHits 
process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = False
#process.famosSimHits.MaterialEffects.NuclearInteraction=false
#process.famosSimHits.MaterialEffects.NuclearInteractionEDM=true
#processfamosSimHits.MaterialEffects.pionEnergies={5.,10.,15.}

# Run what is needed
process.p1 = cms.Path(
    process.offlineBeamSpot+
    process.famosPileUp+
    process.famosSimHits
)

# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['detailedInfo.txt']


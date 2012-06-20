import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos sequences
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("FlatRandomPtGunSource",
    #untracked int32  maxEvents = 1000
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(60.0),
        MinPt = cms.untracked.double(50.0),
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(2.4),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.4),
        MinPhi = cms.untracked.double(-3.14159265359) ## it must be in radians

    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    firstRun = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tracks_Gun_test.root')
)

process.p1 = cms.Path(process.famosWithTracks)
process.outpath = cms.EndPath(process.o1)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.MessageLogger.destinations = ['detailedInfo.txt']



import FWCore.ParameterSet.Config as cms

process = cms.Process("VALID")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos sequences
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

process.load("FastSimulation.Validation.TrackValidation_HighPurity_cff")

#    endpath outpath = { o1 }
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)
process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(10.0),
        MinPt = cms.untracked.double(10.0),
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.0),
        MinPhi = cms.untracked.double(-3.14159265359) ## it must be in radians

    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    firstRun = cms.untracked.uint32(1)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test_pi_10GeV.root')
)

process.p1 = cms.Path(process.famosWithTracks*process.valid)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.multiTrackValidator.outputFile = 'valid_pi_10GeV.root'
process.MessageLogger.destinations = ['detailedInfo_pi10.txt']



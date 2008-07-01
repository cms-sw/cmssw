import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosRecHitProducer")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos Common inputs 
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

# Famos SimHits 
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Magnetic Field (new mapping, 3.8 and 4.0T)
# include "Configuration/StandardSequences/data/MagneticField_38T.cff"
process.load("Configuration.StandardSequences.MagneticField_40T_cff")

process.load("FastSimulation.TrackingRecHitProducer.test.FamosRecHitAnalysis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.5),
        MinE = cms.untracked.double(1.0),
        MinPhi = cms.untracked.double(-3.14159265359), ## it must be in radians

        MaxE = cms.untracked.double(10.0)
    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    firstRun = cms.untracked.uint32(1)
)

process.Output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('rechits.root')
)

process.Path = cms.Path(process.famosWithTrackerHits*process.trackerGSRecHitTranslator*process.FamosRecHitAnalysis)
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.siTrackerGaussianSmearingRecHits.UseCMSSWPixelParametrization = True
process.siTrackerGaussianSmearingRecHits.doRecHitMatching = False
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True



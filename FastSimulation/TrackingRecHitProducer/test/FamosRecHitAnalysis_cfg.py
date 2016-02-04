import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosRecHitAnalysis")

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos Common inputs 
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.GlobalTag.globaltag = "MC_31X_V1::All"
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Magnetic Field (new mapping, 3.8 and 4.0T)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.load("FastSimulation.Configuration.mixNoPU_cfi")
process.mix.playback = cms.untracked.bool(True)
# RecHit Analysis ###
process.load("FastSimulation.TrackingRecHitProducer.FamosRecHitAnalysis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:rechits.root')
)

process.Path = cms.Path(process.mix*process.FamosRecHitAnalysis)



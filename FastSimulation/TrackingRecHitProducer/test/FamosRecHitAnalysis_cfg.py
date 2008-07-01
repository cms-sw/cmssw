import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosRecHitAnalysis")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos Common inputs 
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

# Magnetic Field (new mapping, 3.8 and 4.0T)
# include "Configuration/StandardSequences/data/MagneticField_38T.cff"
process.load("Configuration.StandardSequences.MagneticField_40T_cff")

# RecHit Analysis ###
process.load("FastSimulation.TrackingRecHitProducer.test.FamosRecHitAnalysis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:./rechits.root')
)

process.Path = cms.Path(process.FamosRecHitAnalysis)
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.FamosRecHitAnalysis.UseCMSSWPixelParametrization = False



import FWCore.ParameterSet.Config as cms

process = cms.Process("ana")
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.GlobalTag.globaltag = "IDEAL_30X::All"

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("FastSimulation.TrackingRecHitProducer.GSRecHitValidation_cfi")

process.source = cms.Source("PoolSource",
                            'file:Your_File.root'
                            )



process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.p = cms.Path(process.testanalyzer)
process.MessageLogger.destinations = ['detailedInfo.txt']




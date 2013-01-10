import FWCore.ParameterSet.Config as cms

process = cms.Process("ana")
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.mixNoPU_cfi")

process.GlobalTag.globaltag = "MC_31X_V2::All"


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(15)
    )

process.load("FastSimulation.TrackingRecHitProducer.GSRecHitValidation_cfi")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/MC_31X_V2_FastSim-v1/0002/D2EFE4C6-E16B-DE11-96D1-001D09F28E80.root'
    )
                            )

#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.p = cms.Path(process.mix*process.testanalyzer)
process.MessageLogger.destinations = ['detailedInfo.txt']




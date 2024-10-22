# EGM skimmer
# Author: Rafael Lopes de Sa

import FWCore.ParameterSet.Config as cms

# Run with the 2017 detector
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('SKIM',Run2_2017)

# Import the standard packages for reconstruction and digitization
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoEgamma.EgammaMCTools.pfClusterMatchedToPhotonsSelector_cfi')

# Global Tag configuration ... just using the same as in the RelVal
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '81X_upgrade2017_realistic_v26', '')

process.MessageLogger.cerr.threshold = 'ERROR'
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.options = cms.untracked.PSet( allowUnscheduled = cms.untracked.bool(True) )

# This is where users have some control.
# Define which collections to save and which dataformat we are using
savedCollections = cms.untracked.vstring('drop *',
# The commented ones are large collections that can be kept for debug
#                                         'keep EcalRecHitsSorted_*_*_*',
#                                         'keep recoPFClusters_*_*_*',
#                                         'keep recoCaloClusters_*_*_*',
#                                         'keep recoSuperClusters_*_*_*', 
#                                         'keep recoGsfElectron*_*_*_*',
#                                         'keep recoPhoton*_*_*_*',
#                                         'keep *_mix_MergedTrackTruth_*',
                                         'keep *_reducedEcalRecHits*_*_*',
                                         'keep double_fixedGridRho*_*_*',
                                         'keep recoGenParticles_*_*_*',
                                         'keep GenEventInfoProduct_*_*_*',
                                         'keep PileupSummaryInfos_*_*_*',
                                         'keep *_ecalDigis_*_*',
                                         'keep *_offlinePrimaryVertices_*_*',
                                         'keep *_particleFlowCluster*_*_*')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(15))

process.source = cms.Source("PoolSource",                 
                            fileNames = cms.untracked.vstring(
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/AODSIM/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/005AB6CE-27ED-E611-98CA-E0071B7A8590.root'
        ),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/0416D6B7-04ED-E611-B342-E0071B7A8550.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/14829DD8-04ED-E611-8049-A0000420FE80.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/54AFE9C4-04ED-E611-952D-A0000420FE80.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/5A32C6B9-04ED-E611-B1EB-E0071B7A8550.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/60E162B8-04ED-E611-898D-E0071B7A58F0.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/6A47DD1A-FEEC-E611-81EB-A0000420FE80.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/92B923B6-04ED-E611-9DC9-24BE05C48821.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/B40E77B4-04ED-E611-9E30-E0071B7A45D0.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/C48157B5-04ED-E611-BEC1-E0071B7A45D0.root',
        '/store/mc/PhaseIFall16DR/GluGluHToGG_M-125_13TeV_powheg_pythia8/GEN-SIM-RAW/FlatPU28to62HcalNZSRAW_81X_upgrade2017_realistic_v26-v1/100000/CAED3A16-FEEC-E611-8262-24BE05CEFB41.root'
)
                            )
process.PFCLUSTERoutput = cms.OutputModule("PoolOutputModule",
                                           dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO'),
                                                                        filterName = cms.untracked.string('')
                                                                        ),
                                           eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                           fileName = cms.untracked.string('skimEGMobjects_fromRAW.root'),
                                           outputCommands = savedCollections,
                                           splitLevel = cms.untracked.int32(0)
                                           )

# Run the digitizer to make the trackingparticles
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
process.trackingtruth_step = cms.Path(process.pdigi_valid)

# Remake the PFClusters
process.pfclusters_step = cms.Path(process.bunchSpacingProducer *
                                   process.ecalDigis * 
                                   process.ecalPreshowerDigis * 
                                   process.ecalPreshowerRecHit *
                                   process.ecalMultiFitUncalibRecHit *
                                   process.ecalDetIdToBeRecovered *
                                   process.ecalRecHit *
                                   process.particleFlowRecHitPS * 
                                   process.particleFlowRecHitECAL * 
                                   process.particleFlowClusterECALUncorrected * 
                                   process.particleFlowClusterPS *
                                   process.particleFlowClusterECAL)

# Select the PFClusters we want to calibrate
process.particleFlowClusterECALMatchedToPhotons = process.pfClusterMatchedToPhotonsSelector.clone()
process.selection_step = cms.Path(process.particleFlowClusterECALMatchedToPhotons)

# Ends job and writes our output
process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.PFCLUSTERoutput)

# Schedule definition, rebuilding rechits
process.schedule = cms.Schedule(process.trackingtruth_step,process.pfclusters_step,process.selection_step,process.endjob_step,process.output_step)



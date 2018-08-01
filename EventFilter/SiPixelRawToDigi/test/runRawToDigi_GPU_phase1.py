import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('MyDigis',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
   '/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-RECO/94X_upgrade2018_realistic_v5-v1/10000/58452C1E-A7C8-E711-A56A-0025905A60CE.root',
   #'/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-RECO/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/22E2A744-E3BA-E711-A1A8-5065F3815241.root',
                                                              ),
    secondaryFileNames = cms.untracked.vstring(
   #'/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/62236337-BEBA-E711-9962-4C79BA1810EB.root',
   #'/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/D6384D37-BEBA-E711-B24C-4C79BA180B9F.root',
   '/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/94X_upgrade2018_realistic_v5-v1/10000/0CA1E1A7-9EC8-E711-A286-0025905A612A.root',
   '/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/94X_upgrade2018_realistic_v5-v1/10000/C27D6BA2-9BC8-E711-957F-003048FFD798.root',
   '/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/94X_upgrade2018_realistic_v5-v1/10000/D4FD26AC-99C8-E711-99A0-0CC47A7C3432.root',
   '/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/94X_upgrade2018_realistic_v5-v1/10000/EE36487F-9DC8-E711-9FBD-0CC47A7C3636.root',
                                               )
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVTDEBUGHLTEventContent.outputCommands.append('keep *_*_*_MyDigis')
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:digi.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)


# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelClustersPreSplittingGPU = process.siPixelClustersPreSplitting.clone()
process.siPixelClustersPreSplittingGPU.src = "siPixelDigisGPU"
process.siPixelClustersGPU = process.siPixelClusters.clone()
process.siPixelClustersGPU.pixelClusters = "siPixelClustersPreSplittingGPU"
process.siPixelRecHitsGPU = process.siPixelRecHits.clone()
process.siPixelRecHitsGPU.src = "siPixelClustersGPU"
process.siPixelRecHitsPreSplittingGPU = process.siPixelRecHitsPreSplitting.clone()
process.siPixelRecHitsPreSplittingGPU.src = "siPixelClustersPreSplittingGPU"

### Digi validation

from Validation.SiPixelPhase1DigisV.SiPixelPhase1DigisV_cfi import *

SiPixelPhase1DigisADCGPU = SiPixelPhase1DigisADC.clone()
SiPixelPhase1DigisNdigisGPU = SiPixelPhase1DigisNdigis.clone()
SiPixelPhase1DigisRowsGPU = SiPixelPhase1DigisRows.clone()
SiPixelPhase1DigisColumnsGPU = SiPixelPhase1DigisColumns.clone()
SiPixelPhase1DigisADCGPU.topFolderName = "PixelPhase1V/DigisGPU"
SiPixelPhase1DigisNdigisGPU.topFolderName = "PixelPhase1V/DigisGPU"
SiPixelPhase1DigisRowsGPU.topFolderName = "PixelPhase1V/DigisGPU"
SiPixelPhase1DigisColumnsGPU.topFolderName = "PixelPhase1V/DigisGPU"
SiPixelPhase1DigisColumns.range_max = 450
SiPixelPhase1DigisColumns.range_nbins = 450
SiPixelPhase1DigisColumnsGPU.range_max = 450
SiPixelPhase1DigisColumnsGPU.range_nbins = 450
SiPixelPhase1DigisConfGPU = cms.VPSet(SiPixelPhase1DigisADCGPU,
                                      SiPixelPhase1DigisNdigisGPU,
                                      SiPixelPhase1DigisRowsGPU,
                                      SiPixelPhase1DigisColumnsGPU)
process.SiPixelPhase1DigisAnalyzerVGPU = process.SiPixelPhase1DigisAnalyzerV.clone()
process.SiPixelPhase1DigisHarvesterVGPU = process.SiPixelPhase1DigisHarvesterV.clone()
process.SiPixelPhase1DigisAnalyzerVGPU.src = cms.InputTag("siPixelDigisGPU")
process.SiPixelPhase1DigisAnalyzerVGPU.histograms = SiPixelPhase1DigisConfGPU
process.SiPixelPhase1DigisHarvesterVGPU.histograms = SiPixelPhase1DigisConfGPU

### Cluster validation

from Validation.SiPixelPhase1TrackClustersV.SiPixelPhase1TrackClustersV_cfi import *

SiPixelPhase1TrackClustersChargeGPU = SiPixelPhase1TrackClustersCharge.clone()
SiPixelPhase1TrackClustersSizeXGPU = SiPixelPhase1TrackClustersSizeX.clone()
SiPixelPhase1TrackClustersSizeYGPU = SiPixelPhase1TrackClustersSizeY.clone()
SiPixelPhase1TrackClustersChargeGPU.topFolderName = "PixelPhase1V/ClustersGPU"
SiPixelPhase1TrackClustersSizeXGPU.topFolderName = "PixelPhase1V/ClustersGPU"
SiPixelPhase1TrackClustersSizeYGPU.topFolderName = "PixelPhase1V/ClustersGPU"
SiPixelPhase1TrackClustersConfGPU = cms.VPSet(
                                           SiPixelPhase1TrackClustersChargeGPU,
                                           SiPixelPhase1TrackClustersSizeXGPU,
                                           SiPixelPhase1TrackClustersSizeYGPU
                                           )
process.SiPixelPhase1TrackClustersAnalyzerVGPU = process.SiPixelPhase1TrackClustersAnalyzerV.clone()
process.SiPixelPhase1TrackClustersHarvesterVGPU = process.SiPixelPhase1TrackClustersHarvesterV.clone()
process.SiPixelPhase1TrackClustersAnalyzerVGPU.clusters = "siPixelClustersGPU"
process.SiPixelPhase1TrackClustersAnalyzerVGPU.histograms = SiPixelPhase1TrackClustersConfGPU
process.SiPixelPhase1TrackClustersHarvesterVGPU.histograms = SiPixelPhase1TrackClustersConfGPU

### RecHit validation

from Validation.SiPixelPhase1RecHitsV.SiPixelPhase1RecHitsV_cfi import *

SiPixelPhase1RecHitsInTimeEventsGPU = SiPixelPhase1RecHitsInTimeEvents.clone()
SiPixelPhase1RecHitsOutTimeEventsGPU = SiPixelPhase1RecHitsOutTimeEvents.clone()
SiPixelPhase1RecHitsNSimHitsGPU = SiPixelPhase1RecHitsNSimHits.clone()
SiPixelPhase1RecHitsPosXGPU = SiPixelPhase1RecHitsPosX.clone()
SiPixelPhase1RecHitsPosYGPU = SiPixelPhase1RecHitsPosY.clone()
SiPixelPhase1RecHitsResXGPU = SiPixelPhase1RecHitsResX.clone()
SiPixelPhase1RecHitsResYGPU = SiPixelPhase1RecHitsResY.clone()
SiPixelPhase1RecHitsErrorXGPU = SiPixelPhase1RecHitsErrorX.clone()
SiPixelPhase1RecHitsErrorYGPU = SiPixelPhase1RecHitsErrorY.clone()
SiPixelPhase1RecHitsPullXGPU = SiPixelPhase1RecHitsPullX.clone()
SiPixelPhase1RecHitsPullYGPU = SiPixelPhase1RecHitsPullY.clone()

SiPixelPhase1RecHitsInTimeEventsGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsOutTimeEventsGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsNSimHitsGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsPosXGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsPosYGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsResXGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsResYGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsErrorXGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsErrorYGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsPullXGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsPullYGPU.topFolderName = "PixelPhase1V/RecHitsGPU"
SiPixelPhase1RecHitsConfGPU = cms.VPSet(
                                     SiPixelPhase1RecHitsInTimeEventsGPU,
                                     SiPixelPhase1RecHitsOutTimeEventsGPU,
                                     SiPixelPhase1RecHitsNSimHitsGPU,
                                     SiPixelPhase1RecHitsPosXGPU,
                                     SiPixelPhase1RecHitsPosYGPU,
                                     SiPixelPhase1RecHitsResXGPU,
                                     SiPixelPhase1RecHitsResYGPU,
                                     SiPixelPhase1RecHitsErrorXGPU,
                                     SiPixelPhase1RecHitsErrorYGPU,
                                     SiPixelPhase1RecHitsPullXGPU,
                                     SiPixelPhase1RecHitsPullYGPU,
                                     )
process.SiPixelPhase1RecHitsAnalyzerVGPU = process.SiPixelPhase1RecHitsAnalyzerV.clone()
process.SiPixelPhase1RecHitsHarvesterVGPU = process.SiPixelPhase1RecHitsHarvesterV.clone()
process.SiPixelPhase1RecHitsAnalyzerVGPU.src = "siPixelRecHitsGPU"
process.SiPixelPhase1RecHitsAnalyzerVGPU.histograms = SiPixelPhase1RecHitsConfGPU
process.SiPixelPhase1RecHitsHarvesterVGPU.histograms = SiPixelPhase1RecHitsConfGPU

process.InitialStepPreSplitting = cms.Sequence(
                                                #process.trackerClusterCheckPreSplitting
                                               #+process.initialStepSeedLayersPreSplitting
                                               #+process.initialStepTrackingRegionsPreSplitting
                                               #+process.initialStepHitDoubletsPreSplitting
                                               #+process.initialStepHitQuadrupletsPreSplitting
                                               #+process.initialStepSeedsPreSplitting
                                               #+process.initialStepTrackCandidatesPreSplitting
                                               process.initialStepTracksPreSplitting
                                               +process.firstStepPrimaryVerticesPreSplitting
                                               +process.caloTowerForTrkPreSplitting
                                               +process.ak4CaloJetsForTrkPreSplitting
                                               +process.jetsForCoreTrackingPreSplitting
                                               #+process.initialStepTrackRefsForJetsPreSplitting
                                               #+process.caloTowerForTrkPreSplitting
                                               #+process.ak4CaloJetsForTrkPreSplitting
                                               #+process.jetsForCoreTrackingPreSplitting
#                                               +process.siPixelClusters
#                                               +process.siPixelRecHits
#                                               +process.MeasurementTrackerEvent
#                                               +process.siPixelClusterShapeCache
                                               )

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siPixelDigis)
process.raw2digiGPU_step = cms.Path(process.siPixelDigisGPU)
process.clustering = cms.Path(process.InitialStepPreSplitting + process.siPixelClustersPreSplittingGPU + process.siPixelClustersGPU)
process.rechits = cms.Path(process.siPixelRecHitsGPU + process.siPixelRecHitsPreSplittingGPU)

process.validation_step = cms.Path(process.SiPixelPhase1DigisAnalyzerV + process.SiPixelPhase1TrackClustersAnalyzerV + process.SiPixelPhase1RecHitsAnalyzerV)
process.harvesting_step = cms.Path(process.SiPixelPhase1DigisHarvesterV + process.SiPixelPhase1TrackClustersHarvesterV + process.SiPixelPhase1RecHitsHarvesterV)

process.validationGPU_step = cms.Path(process.SiPixelPhase1DigisAnalyzerVGPU + process.SiPixelPhase1TrackClustersAnalyzerVGPU + process.SiPixelPhase1RecHitsAnalyzerVGPU)
process.harvestingGPU_step = cms.Path(process.SiPixelPhase1DigisHarvesterVGPU + process.SiPixelPhase1TrackClustersHarvesterVGPU + process.SiPixelPhase1RecHitsHarvesterVGPU)
process.RECOSIMoutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(
#                                process.raw2digi_step,
                                process.raw2digiGPU_step,
                                process.clustering,
                                process.rechits,
                                process.validation_step,
                                process.harvesting_step,
                                process.validationGPU_step,
                                process.harvestingGPU_step,
                                process.RECOSIMoutput_step,
                                process.DQMoutput_step,
                                process.dqmsave_step
                                )


# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion



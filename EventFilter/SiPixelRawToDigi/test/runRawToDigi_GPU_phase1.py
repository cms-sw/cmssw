import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('MyDigis',eras.Run2_2017)

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
                                                              'file:step2.root',
#   '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-RECO/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/22E2A744-E3BA-E711-A1A8-5065F3815241.root',
                                                              ),
    secondaryFileNames = cms.untracked.vstring(
#   '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/62236337-BEBA-E711-9962-4C79BA1810EB.root',
#   '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/D6384D37-BEBA-E711-B24C-4C79BA180B9F.root',
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

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:digi.root'),
#    outputCommands = cms.untracked.vstring("drop *", "keep *_simSiPixelDigis_*_*", "keep *_siPixelDigisGPU_*_*"),
    outputCommands = cms.untracked.vstring("keep *"),
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_design', '')

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelClustersPreSplitting.src = cms.InputTag("siPixelDigisGPU")

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


# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siPixelDigis)
process.raw2digiGPU_step = cms.Path(process.siPixelDigisGPU)
process.clustering = cms.Path(process.siPixelClustersPreSplitting)
process.validation_step = cms.Path(process.SiPixelPhase1DigisAnalyzerV + process.SiPixelPhase1DigisAnalyzerVGPU)
process.harvesting_step = cms.Path(process.SiPixelPhase1DigisHarvesterV + process.SiPixelPhase1DigisHarvesterVGPU)
process.RECOSIMoutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(#process.raw2digi_step,
                                process.raw2digiGPU_step,
#                                process.clustering,
                                process.validation_step,
                                process.harvesting_step,
                                process.RECOSIMoutput_step,
                                process.DQMoutput_step,
                                process.dqmsave_step)


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



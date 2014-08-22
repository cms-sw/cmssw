# Auto generated configuration file
# using: 
# Revision: 1.372.2.22 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: harvest -s HARVESTING:validationHarvesting --mc --conditions auto:startup_GRun --harvesting AtJobEnd --filein file:step2_RAW2DIGI_RECO_VALIDATION.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_2012_Summer_50ns_PoissonOOTPU_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtJobEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:step2_RAW2DIGI_RECO_VALIDATION.root'),
    processingMode = cms.untracked.string('RunsAndLumis')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('harvest nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup_GRun', '')

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod)
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.validationHarvestingFS = cms.Path(process.HarvestingFastSim)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.alcaHarvesting = cms.Path()
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,process.validationHarvesting,process.dqmsave_step)


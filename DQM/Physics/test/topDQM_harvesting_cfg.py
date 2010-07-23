# Auto generated configuration file
# using: 
# Revision: 1.169 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3 -s HARVESTING:dqmHarvesting --harvesting AtRunEnd --conditions FrontierConditions_GlobalTag,GR10_P_V5::All --filein file:step2_DQM.root --data --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('step3 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
"file:topDQM_production_10_ref.root"),
    processingMode = cms.untracked.string('RunsAndLumis')
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'START38_V7::All'
#process.GlobalTag.globaltag = 'GR10_P_V5::All'

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationpreprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_preprod)
process.validationprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_prod)
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep*process.DQMOffline_Certification)
process.validationHarvesting = cms.Path(process.postValidation*process.hltpostvalidation)
process.validationHarvestingFS = cms.Path(process.HarvestingFastSim)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.dqmsave_step = cms.Path(process.DQMSaver)

process.DQMStore.collateHistograms = True
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = True
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/TopVal/CMSSW_3_6_1/RECO')
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,process.dqmsave_step)

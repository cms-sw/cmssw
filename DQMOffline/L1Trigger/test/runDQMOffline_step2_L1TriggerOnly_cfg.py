# Auto generated configuration file
# using:
# Revision: 1.1
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: test_11_b_1 -s HARVESTING:dqmHarvesting --conditions auto:com10 --data --filein file:test_11_a_1_RAW2DIGI_RECO_DQM.root --scenario pp --customise DQMTools/Tests/customHarvesting.py --no_exec --python_filename=test_11_b_1.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Removing other DQM modules form the DQMOffline_SecondStep_PreDPG
process.DQMOffline_SecondStep_PreDPG.remove(process.dqmDcsInfoClient)
process.DQMOffline_SecondStep_PreDPG.remove(process.ecal_dqm_client_offline)
process.DQMOffline_SecondStep_PreDPG.remove(process.hcalOfflineDQMClient)
process.DQMOffline_SecondStep_PreDPG.remove(process.SiStripOfflineDQMClient)
process.DQMOffline_SecondStep_PreDPG.remove(process.PixelOfflineDQMClientNoDataCertification)
process.DQMOffline_SecondStep_PreDPG.remove(process.dtClients)
process.DQMOffline_SecondStep_PreDPG.remove(process.rpcTier0Client)
process.DQMOffline_SecondStep_PreDPG.remove(process.cscOfflineCollisionsClients)
process.DQMOffline_SecondStep_PreDPG.remove(process.es_dqm_client_offline)
process.DQMOffline_SecondStep_PreDPG.remove(process.dqmFEDIntegrityClient)

# Removing other DQM modules form the DQMOffline_SecondStep_PrePOG
process.DQMOffline_SecondStep_PrePOG.remove(process.muonQualityTests)
process.DQMOffline_SecondStep_PrePOG.remove(process.egammaPostProcessing)
#process.DQMOffline_SecondStep_PrePOG.remove(process.l1TriggerDqmOfflineClient)
process.DQMOffline_SecondStep_PrePOG.remove(process.triggerOfflineDQMClient)
process.DQMOffline_SecondStep_PrePOG.remove(process.hltOfflineDQMClient)
process.DQMOffline_SecondStep_PrePOG.remove(process.bTagCollectorSequence)
process.DQMOffline_SecondStep_PrePOG.remove(process.alcaBeamMonitorClient)
process.DQMOffline_SecondStep_PrePOG.remove(process.SusyPostProcessorSequence)
process.DQMOffline_SecondStep_PrePOG.remove(process.runTauEff)

# Removing other DQM modules form the DQMOffline_Certification
process.DQMOffline_Certification.remove(process.daq_dqmoffline)
process.DQMOffline_Certification.remove(process.dcs_dqmoffline)
process.DQMOffline_Certification.remove(process.crt_dqmoffline)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:L1TOffline_L1TriggerOnly_job1_RAW2DIGI_RECO_DQM.root'),
    processingMode = cms.untracked.string('RunsAndLumis')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('L1TOffline_L1TriggerOnly_Harvested.root nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod)
process.validationHarvesting = cms.Path(process.postValidation+process.hltpostvalidation)
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.validationHarvestingFS = cms.Path(process.HarvestingFastSim)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.alcaHarvesting = cms.Path()
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,process.dqmHarvesting,process.dqmsave_step)

# customisation of the process.

# Automatic addition of the customisation function from DQMTools.Tests.customHarvesting
from DQMTools.Tests.customHarvesting import customise

#call to customisation function customise imported from DQMTools.Tests.customHarvesting
process = customise(process)

# End of customisation functions

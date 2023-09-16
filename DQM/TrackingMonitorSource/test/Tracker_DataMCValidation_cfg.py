# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 -s DQM -n 1 --eventcontent DQM --conditions auto:com10 --filein /store/relval/CMSSW_7_1_2/MinimumBias/RECO/GR_R_71_V7_dvmc_RelVal_mb2012Cdvmc-v1/00000/00209DF4-3708-E411-9FA7-0025905A6126.root --data --no_exec --python_filename=test_step1_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('DQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
  secondaryFileNames = cms.untracked.vstring(),
  fileNames = cms.untracked.vstring([
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/c02ca5ba-f454-4cd3-b114-b55e0309f9db.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/74379440-0e40-49dc-aaa1-0158c3589590.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/6a763153-0010-4a49-b737-0345521f2768.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/a6ffe171-273f-4876-8de2-984da92c34e1.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/4bc3f9a0-1c1e-4e24-8caa-63a5046ee987.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/d2eddcbd-9a36-4266-a345-9b318a97c666.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/8c2b94e6-436d-4464-9713-041e32d61fc0.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/4cc07559-176e-436b-b452-710d309b3f34.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/b25d547a-5783-4865-966d-05b20462ccc3.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/57301d29-4c60-442f-a9f9-966152b845fd.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/9715ac43-30b1-4077-9e51-94c594bbdf41.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/dd5118f8-c9b8-4c22-b222-a55100469080.root',
      '/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/f07c9474-56cb-4b51-a47e-779cf143548c.root'])
)

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step1 nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('step1_DQM_1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '132X_mcRun3_2023_realistic_v2', '')

# Tracker Data MC validation suite
process.load('DQM.TrackingMonitorSource.TrackingDataMCValidation_Standalone_cff')
process.analysis_step = cms.Path(process.standaloneValidationElec)
# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step, process.endjob_step, process.DQMoutput_step)

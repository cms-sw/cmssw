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
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
  secondaryFileNames = cms.untracked.vstring(),
  fileNames = cms.untracked.vstring([
       '/store/relval/CMSSW_7_4_15/RelValNuGun_UP15/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2-v1/00000/0AEF9C14-1A72-E511-93A4-0025905A60E0.root',
       '/store/relval/CMSSW_7_4_15/RelValNuGun_UP15/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2-v1/00000/0C0F2713-1A72-E511-B602-0025905A48EC.root',
       '/store/relval/CMSSW_7_4_15/RelValNuGun_UP15/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2-v1/00000/1C089D5B-1B72-E511-962F-0025905A612C.root',
       '/store/relval/CMSSW_7_4_15/RelValNuGun_UP15/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2-v1/00000/2024325B-1B72-E511-AE70-0025905B85EE.root',
       '/store/relval/CMSSW_7_4_15/RelValNuGun_UP15/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2-v1/00000/7A290461-1B72-E511-8422-0026189438CE.root' 
  ])
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
    fileName = cms.untracked.string('/afs/cern.ch/work/k/kmondal/public/DatavsMC/RENovember2015/CMSSW_7_4_15_patch1/src/DQM/TrackingMonitorSource/test/Jobs/MC/MinBias/step1_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '74X_mcRun2_asymptotic_v2', '')

# Tracker Data MC validation suite
process.load('DQM.TrackingMonitorSource.TrackingDataMCValidation_Standalone_cff')
process.analysis_step = cms.Path(process.standaloneValidationMinbias)

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step, process.endjob_step, process.DQMoutput_step)

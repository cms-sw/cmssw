import FWCore.ParameterSet.Config as cms
import os
from Configuration.StandardSequences.Eras import eras

process = cms.Process('L1TStage2EmulatorDQM', eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load(
    'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    #input=cms.untracked.int32(50)
)

# Input source

# das_client.py --limit 0 --query "file dataset=/RelValTTbarLepton_13/CMSSW_8_1_0_pre12-81X_mcRun2_asymptotic_v8-v1/GEN-SIM-RECO" > fileList.global
# download first file
# export $xrdfile=`head -1 fileList.global`
# xrdcp root://xrootd-cms.infn.it/$f TEST.root
# echo "file://$PWD/TEST.root" > fileList.local
# or use fileList.global
with open('fileList.local') as f:
    fileList = f.readlines()
# das_client.py --limit 0 --query "file dataset=/RelValTTbarLepton_13/CMSSW_8_1_0_pre12-81X_mcRun2_asymptotic_v8-v1/GEN-SIM-DIGI-RAW-HLTDEBUG" > fileListRAW.global
# export xrdfile=`head -1 fileListRAW.global`
# xrdcp root://xrootd-cms.infn.it/$xrdfile TEST_RAW.root
# echo "file://$PWD/TEST_RAW.root" > fileListRAW.local
# or use fileListRAW.global
with open('fileListRAW.local') as f:
    fileListRAW = f.readlines()
process.source = cms.Source(
    "PoolSource",
#     fileNames=cms.untracked.vstring(fileList[0]),
#     secondaryFileNames=cms.untracked.vstring(fileListRAW),
    fileNames=cms.untracked.vstring(
        '/store/data/Run2016H/SingleMuon/RAW-RECO/MuTau-PromptReco-v2/000/282/092/00001/B89AA967-FB8A-E611-9A09-FA163E845EAD.root'),
        #'file:///vagrant/workspace/DQMOffline/src/doubleEG/009E5CBE-AE87-E611-9122-0025905A60C6.root'),
)

process.options = cms.untracked.PSet(

)

# Output definition
process.DQMoutput = cms.OutputModule(
    "DQMRootOutputModule",
    fileName=cms.untracked.string(
        "L1TOffline_L1TStage2CaloLayer2_job1_RAW2DIGI_RECO_DQM.root")
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)

process.load('DQMOffline.L1Trigger.L1TStage2CaloLayer2Offline_cfi')
process.load('DQMOffline.L1Trigger.L1TEGammaOffline_cfi')
process.load('DQMOffline.L1Trigger.L1TTauOffline_cfi')

if os.environ.get('DEBUG', False):
    process.MessageLogger.cout.threshold=cms.untracked.string('DEBUG')
    process.MessageLogger.debugModules = cms.untracked.vstring(
        '*',
    )

process.dqmoffline_step = cms.Path(
    process.l1tStage2CaloLayer2OfflineDQMEmu +
    process.l1tStage2CaloLayer2OfflineDQM +
    process.l1tEGammaOfflineDQM +
    process.l1tEGammaOfflineDQMEmu +
    process.l1tTauOfflineDQM +
    process.l1tTauOfflineDQMEmu
)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
# Schedule definition
process.schedule = cms.Schedule(
    process.raw2digi_step,
)

# customisation of the process.

# Automatic addition of the customisation function from
# L1Trigger.Configuration.customiseReEmul
from L1Trigger.Configuration.customiseReEmul import L1TReEmulFromRAW

# call to customisation function L1TReEmulFromRAW imported from
# L1Trigger.Configuration.customiseReEmul
process = L1TReEmulFromRAW(process)
process.schedule.append(process.dqmoffline_step)
process.schedule.append(process.DQMoutput_step)

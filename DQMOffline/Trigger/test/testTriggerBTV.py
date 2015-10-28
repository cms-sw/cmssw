import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMOffline.Trigger.BTVHLTOfflineSource_cfi")
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = '74X_dataRun2_Prompt_v0'
process.prefer("GlobalTag")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/3EC1C553-BF36-E511-9BF4-00259059649C.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/A2489ECC-BE36-E511-88CB-002618943900.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/B40AE4FD-B436-E511-A272-003048FFD7A2.root' 
    ),
    secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/08BAC960-B436-E511-8306-0025905964C4.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/EA8F0E28-AA36-E511-9539-003048FFCBB0.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/483551C9-A936-E511-9A3E-002618943910.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/7603F5AF-B436-E511-AC18-0025905A610C.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/CE647DAD-B136-E511-BC87-0025905B8592.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/E2E857AB-B436-E511-A4EA-0025905A6138.root',
       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/089CF124-AA36-E511-A05F-0025905A605E.root',
    ),
)

process.p = cms.EndPath(
process.BTVHLTOfflineSource + process.dqmEnv+process.dqmSaver
)

process.DQMStore.verbose = 0
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

outfile = open('config.py','w')
print >> outfile,process.dumpPython()
outfile.close()

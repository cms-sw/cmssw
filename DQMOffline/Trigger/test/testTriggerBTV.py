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
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/3EC1C553-BF36-E511-9BF4-00259059649C.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/A2489ECC-BE36-E511-88CB-002618943900.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v2-v1/00000/B40AE4FD-B436-E511-A272-003048FFD7A2.root' 
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v4-v1/00000/1A8546C9-5569-E511-A50B-0026189438F7.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v4-v1/00000/82BFCA1B-4669-E511-87AA-0025905A60B8.root',
       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v4-v1/00000/84BCCDC8-5569-E511-94EB-002618943854.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/2ABA327E-2969-E511-A5A0-003048FFD736.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/46C054AB-2B69-E511-8CD3-0025905A60EE.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/5A3A6CFA-2769-E511-95AE-0025905B85A2.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/829C9381-3069-E511-943E-002590596484.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/84A3E212-3869-E511-A485-0025905938AA.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/8C278DF6-2769-E511-9A75-0025905A60BC.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/986B93AB-2B69-E511-8555-0025905A60EE.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/9A8902F8-2769-E511-A3AB-0025905B85EE.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/CE0D06A7-2B69-E511-A2F2-0025905964B4.root',
       'file:///afs/cern.ch/user/s/sdonato/eos/cms/store/relval/CMSSW_7_6_0_pre6/RelValTTbarLepton_13/GEN-SIM-DIGI-RAW-HLTDEBUG/76X_mcRun2_asymptotic_v4-v1/00000/D4F42F6F-2969-E511-AC2A-00261894390C.root' 
       #       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/08BAC960-B436-E511-8306-0025905964C4.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/EA8F0E28-AA36-E511-9539-003048FFCBB0.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/483551C9-A936-E511-9A3E-002618943910.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/7603F5AF-B436-E511-AC18-0025905A610C.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/CE647DAD-B136-E511-BC87-0025905B8592.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/E2E857AB-B436-E511-A4EA-0025905A6138.root',
#       '/store/relval/CMSSW_7_6_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/089CF124-AA36-E511-A05F-0025905A605E.root',
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

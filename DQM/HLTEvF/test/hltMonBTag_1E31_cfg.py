import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HLTEvF.hltMonBTagIPSource_cfi")
process.hltMonBTagIPSource.pathName   = 'HLT_BTagIP_Jet80'
process.hltMonBTagIPSource.L1Filter   = 'hltL1sBTagIPJet80'
process.hltMonBTagIPSource.L2Filter   = 'hltBJet80'
process.hltMonBTagIPSource.L2Jets     = 'hltMCJetCorJetIcone5Regional'
process.hltMonBTagIPSource.L25TagInfo = 'hltBLifetimeL25TagInfosStartup'
process.hltMonBTagIPSource.L25JetTags = 'hltBLifetimeL25BJetTagsStartup'
process.hltMonBTagIPSource.L25Filter  = 'hltBLifetimeL25FilterStartup'
process.hltMonBTagIPSource.L3TagInfo  = 'hltBLifetimeL3TagInfosStartup'
process.hltMonBTagIPSource.L3JetTags  = 'hltBLifetimeL3BJetTagsStartup'
process.hltMonBTagIPSource.L3Filter   = 'hltBLifetimeL3FilterStartup'
process.hltMonBTagIPSource.storeROOT  = True

process.load("DQM.HLTEvF.hltMonBTagMuSource_cfi")
process.hltMonBTagMuSource.pathName   = 'HLT_BTagMu_Jet20'
process.hltMonBTagMuSource.L1Filter   = 'hltL1sBTagMuJet20'
process.hltMonBTagMuSource.L2Filter   = 'hltBJet20'
process.hltMonBTagMuSource.L2Jets     = 'hltMCJetCorJetIcone5'
process.hltMonBTagMuSource.L25TagInfo = 'hltBSoftMuonL25TagInfos'
process.hltMonBTagMuSource.L25JetTags = 'hltBSoftMuonL25BJetTagsByDR'
process.hltMonBTagMuSource.L25Filter  = 'hltBSoftMuonL25FilterByDR'
process.hltMonBTagMuSource.L3TagInfo  = 'hltBSoftMuonL3TagInfos'
process.hltMonBTagMuSource.L3JetTags  = 'hltBSoftMuonL3BJetTagsByDR'
process.hltMonBTagMuSource.L3Filter   = 'hltBSoftMuonL3FilterByDR'
process.hltMonBTagMuSource.storeROOT  = True

process.load("DQM.HLTEvF.hltMonBTagIPClient_cfi")
process.hltMonBTagIPSource.pathName  = 'HLT_BTagIP_Jet80'
process.hltMonBTagIPClient.storeROOT = True

process.load("DQM.HLTEvF.hltMonBTagMuClient_cfi")
process.hltMonBTagMuSource.pathName   = 'HLT_BTagMu_Jet20'
process.hltMonBTagMuClient.storeROOT  = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# /RelValTTbar/CMSSW_3_3_0_pre2-MC_31X_V8-v1/GEN-SIM-DIGI-RAW-HLTDEBUG
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0003/76F5340B-469C-DE11-BE34-0018F3D095EC.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/F8FD4ADF-AE9B-DE11-B22A-001A92810AE4.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/E2BB2692-AF9B-DE11-8FE5-0018F3D0967A.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/B85ADBDC-AE9B-DE11-849F-0018F3D096DA.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/96E80091-AF9B-DE11-8B3E-0018F3D096CA.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/96B152E0-AE9B-DE11-B5AF-0017312B55A3.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/82E133DE-AE9B-DE11-99E3-0017312310E7.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/76BE4E37-B09B-DE11-8AC2-0018F3D09634.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/680CABD8-AE9B-DE11-AC8E-003048679084.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/56A88F91-AF9B-DE11-ADDB-001A92971B38.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/2CDA3229-AE9B-DE11-A4BF-0017312B55A3.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/224B5A27-AE9B-DE11-B35E-0018F3D09658.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/223C7493-AF9B-DE11-9685-0017312B5DA9.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V8-v1/0002/20496790-AF9B-DE11-AFDF-0018F3D0969C.root'
    )
)

process.dqm = cms.Path( process.hltMonBTagIPSource + process.hltMonBTagMuSource  + process.hltMonBTagIPClient + process.hltMonBTagMuClient )

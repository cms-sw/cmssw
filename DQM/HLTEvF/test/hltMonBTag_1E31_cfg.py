import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# /RelValTTbar/CMSSW_3_3_0_pre6-MC_31X_V9-v1/GEN-SIM-DIGI-RAW-HLTDEBUG 
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/F2217C95-C7AF-DE11-B75F-001D09F24763.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/CAFDFC5E-F1AF-DE11-A913-001D09F295FB.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/BA72C65E-C8AF-DE11-A9B6-001D09F244DE.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/9A338609-C7AF-DE11-B9AC-0019B9F72CE5.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/347F47BC-C7AF-DE11-9C96-001D09F2B2CF.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/2E2C46FE-C6AF-DE11-A582-0019B9F70468.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/1E81FB1F-C9AF-DE11-A40F-001D09F24024.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0005/16EC7C8D-C7AF-DE11-BF1E-000423D99AA2.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/F4667131-C2AF-DE11-86E5-001D09F2905B.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/E8BED280-C5AF-DE11-81F5-001D09F251FE.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/B288FF93-C4AF-DE11-8C7C-001D09F242EF.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/B07F4F2C-C4AF-DE11-9EEF-000423D944FC.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/A604A234-C6AF-DE11-BF15-000423D99F1E.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/74D3CB57-C3AF-DE11-BB96-001D09F2462D.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/6E935E9A-C0AF-DE11-A77C-000423D8F63C.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/561C6578-C5AF-DE11-ADCD-0019B9F70468.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/4AD49ABA-C4AF-DE11-9379-001D09F28755.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/28CEED94-BFAF-DE11-BC64-000423D9A2AE.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/105E2C2E-C5AF-DE11-90F6-001D09F28755.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/0C47EFD2-C3AF-DE11-A3B3-001D09F242EF.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0004/06413113-C6AF-DE11-9BD9-000423D98DD4.root'
    )
)

process.load("DQMServices.Core.DQM_cfg")

import DQM.HLTEvF.hltMonBTagIPSource_cfi
import DQM.HLTEvF.hltMonBTagMuSource_cfi
import DQM.HLTEvF.hltMonBTagIPClient_cfi
import DQM.HLTEvF.hltMonBTagMuClient_cfi

# definition of the Sources for 1E31
process.hltMonBTagIP_Jet80_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
process.hltMonBTagIP_Jet80_Source.pathName    = 'HLT_BTagIP_Jet80'
process.hltMonBTagIP_Jet80_Source.L1Filter    = 'hltL1sBTagIPJet80'
process.hltMonBTagIP_Jet80_Source.L2Filter    = 'hltBJet80'
process.hltMonBTagIP_Jet80_Source.L2Jets      = 'hltMCJetCorJetIcone5Regional'
process.hltMonBTagIP_Jet80_Source.L25TagInfo  = 'hltBLifetimeL25TagInfosStartup'
process.hltMonBTagIP_Jet80_Source.L25JetTags  = 'hltBLifetimeL25BJetTagsStartup'
process.hltMonBTagIP_Jet80_Source.L25Filter   = 'hltBLifetimeL25FilterStartup'
process.hltMonBTagIP_Jet80_Source.L3TagInfo   = 'hltBLifetimeL3TagInfosStartup'
process.hltMonBTagIP_Jet80_Source.L3JetTags   = 'hltBLifetimeL3BJetTagsStartup'
process.hltMonBTagIP_Jet80_Source.L3Filter    = 'hltBLifetimeL3FilterStartup'
process.hltMonBTagIP_Jet80_Source.storeROOT   = True

process.hltMonBTagIP_Jet120_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
process.hltMonBTagIP_Jet120_Source.pathName   = 'HLT_BTagIP_Jet120'
process.hltMonBTagIP_Jet120_Source.L1Filter   = 'hltL1sBTagIPJet120'
process.hltMonBTagIP_Jet120_Source.L2Filter   = 'hltBJet120'
process.hltMonBTagIP_Jet120_Source.L2Jets     = 'hltMCJetCorJetIcone5Regional'
process.hltMonBTagIP_Jet120_Source.L25TagInfo = 'hltBLifetimeL25TagInfosStartup'
process.hltMonBTagIP_Jet120_Source.L25JetTags = 'hltBLifetimeL25BJetTagsStartup'
process.hltMonBTagIP_Jet120_Source.L25Filter  = 'hltBLifetimeL25FilterStartup'
process.hltMonBTagIP_Jet120_Source.L3TagInfo  = 'hltBLifetimeL3TagInfosStartup'
process.hltMonBTagIP_Jet120_Source.L3JetTags  = 'hltBLifetimeL3BJetTagsStartup'
process.hltMonBTagIP_Jet120_Source.L3Filter   = 'hltBLifetimeL3FilterStartup'
process.hltMonBTagIP_Jet120_Source.storeROOT  = True

process.hltMonBTagMu_Jet20_Source = DQM.HLTEvF.hltMonBTagMuSource_cfi.hltMonBTagMuSource.clone()
process.hltMonBTagMu_Jet20_Source.pathName    = 'HLT_BTagMu_Jet20'
process.hltMonBTagMu_Jet20_Source.L1Filter    = 'hltL1sBTagMuJet20'
process.hltMonBTagMu_Jet20_Source.L2Filter    = 'hltBJet20'
process.hltMonBTagMu_Jet20_Source.L2Jets      = 'hltMCJetCorJetIcone5'
process.hltMonBTagMu_Jet20_Source.L25TagInfo  = 'hltBSoftMuonL25TagInfos'
process.hltMonBTagMu_Jet20_Source.L25JetTags  = 'hltBSoftMuonL25BJetTagsByDR'
process.hltMonBTagMu_Jet20_Source.L25Filter   = 'hltBSoftMuonL25FilterByDR'
process.hltMonBTagMu_Jet20_Source.L3TagInfo   = 'hltBSoftMuonL3TagInfos'
process.hltMonBTagMu_Jet20_Source.L3JetTags   = 'hltBSoftMuonL3BJetTagsByDR'
process.hltMonBTagMu_Jet20_Source.L3Filter    = 'hltBSoftMuonL3FilterByDR'
process.hltMonBTagMu_Jet20_Source.storeROOT   = True

process.hltMonBTagSource_1E31 = cms.Sequence( process.hltMonBTagIP_Jet80_Source + process.hltMonBTagMu_Jet20_Source + process.hltMonBTagIP_Jet120_Source )

# definition of the Clients for 1E31
process.hltMonBTagIP_Jet80_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
process.hltMonBTagIP_Jet80_Client.pathName    = 'HLT_BTagIP_Jet80'
process.hltMonBTagIP_Jet80_Client.updateRun   = True
process.hltMonBTagIP_Jet80_Client.storeROOT   = True

process.hltMonBTagIP_Jet120_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
process.hltMonBTagIP_Jet120_Client.pathName   = 'HLT_BTagIP_Jet120'
process.hltMonBTagIP_Jet120_Client.updateRun  = True
process.hltMonBTagIP_Jet120_Client.storeROOT  = True

process.hltMonBTagMu_Jet20_Client = DQM.HLTEvF.hltMonBTagMuClient_cfi.hltMonBTagMuClient.clone()
process.hltMonBTagMu_Jet20_Client.pathName    = 'HLT_BTagMu_Jet20'
process.hltMonBTagMu_Jet20_Client.updateRun   = True
process.hltMonBTagMu_Jet20_Client.storeROOT   = True

process.hltMonBTagClient_1E31 = cms.Sequence( process.hltMonBTagIP_Jet80_Client + process.hltMonBTagMu_Jet20_Client + process.hltMonBTagIP_Jet120_Client )

process.dqm = cms.Path( process.hltMonBTagSource_1E31 + process.hltMonBTagClient_1E31 )
# foo bar baz

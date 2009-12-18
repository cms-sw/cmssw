import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.HLTEvF.hltMonBTagIPSource_cfi")
process.hltMonBTagIPSource.storeROOT  = True

process.load("DQM.HLTEvF.hltMonBTagMuSource_cfi")
process.hltMonBTagMuSource.storeROOT = True

process.load("DQM.HLTEvF.hltMonBTagIPClient_cfi")
process.hltMonBTagIPClient.storeROOT = True

process.load("DQM.HLTEvF.hltMonBTagMuClient_cfi")
process.hltMonBTagMuClient.storeROOT = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# /RelValTTbar/CMSSW_3_3_0_pre2-STARTUP31X_V7-v1/GEN-SIM-DIGI-RAW-HLTDEBUG
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0003/A42AA22B-469C-DE11-AB8C-001731AF66F7.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/EED21001-AC9B-DE11-9B19-0030486790FE.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/E4A081AA-AA9B-DE11-9820-003048678FC4.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/AA864C56-AB9B-DE11-97C1-003048679228.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/928905F3-B09B-DE11-ABD3-001A92971AA4.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/7AB5FDF7-A99B-DE11-A1BB-003048678B86.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/6E4DFB6F-B29B-DE11-B7D0-001BFCDBD100.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/66F54318-B39B-DE11-9C6B-0018F3D096DA.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/62B51270-B29B-DE11-80F1-001A92971B92.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/4661A5B5-B39B-DE11-99BC-003048678FEA.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/2E0A4BAB-AA9B-DE11-A370-001A928116DC.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/2A775456-AB9B-DE11-B8B4-00304867920C.root',
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0002/1A3047A7-AA9B-DE11-9DEF-0030486792B6.root'
    )
)

process.dqm = cms.Path( process.hltMonBTagIPSource + process.hltMonBTagMuSource  + process.hltMonBTagIPClient + process.hltMonBTagMuClient )

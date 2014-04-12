import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(
        # TTbar RelVal 2.1.0
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/02F674DE-A160-DD11-A882-001617DBD5AC.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/04327AC0-1C61-DD11-93B8-001BFCDBD19E.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/06621B92-A060-DD11-B33C-000423D6CA6E.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/08059389-0E61-DD11-89D1-001A928116DC.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0830A57C-1561-DD11-9B9D-001731A28A31.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0C6AEA0F-0E61-DD11-BF9F-0018F3D096E0.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/0C74136A-1761-DD11-80D7-0018F3D09686.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/18D5104F-A060-DD11-8746-000423D991D4.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/1ED2C659-1861-DD11-857C-0017312B5F3F.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/1EDB2A98-0D61-DD11-9D88-003048767DDB.root'
           )
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.DQMStore = cms.Service("DQMStore")

#Load the Validation
process.load("HLTriggerOffline.JetMET.Validation.HLTJetMETValidation_cff")

#Pickk your favourite Boson (23 is Z , 25 is H0 , 37 is H+)
#process.JetMETMCProducer.BosonID = 25









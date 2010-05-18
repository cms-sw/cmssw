import FWCore.ParameterSet.Config as cms

process = cms.Process("TopDQM")
process.load("DQM.Physics.topSingleLeptonDQM_cfi")
process.load("DQM.Physics.topDiLeptonOfflineDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Physics/TopSingleLeptonDQM/DataSet')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource"
    ,fileNames = cms.untracked.vstring(
##     '/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/14920B0A-0DE8-DE11-B138-002618943926.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/1AD1F37E-0BE8-DE11-8D83-00261894396A.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/AC476888-0CE8-DE11-8EDC-0026189438D4.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/4ADBBCAE-37E8-DE11-AE89-00304867C1BA.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/6ABDD43B-13E8-DE11-8A47-001A92971BA0.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/744B08B2-12E8-DE11-A729-001A928116B8.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/A2CC4B57-11E8-DE11-B413-003048678D9A.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/B69516B8-12E8-DE11-982F-00304867BFAE.root'
##    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/CEFA8143-12E8-DE11-A51F-0018F3D096E4.root' 

      '/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/1A19B479-BA3A-DF11-8E43-0017A4770410.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/1A2CED78-BA3A-DF11-98CD-0017A4771010.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/3AE61B7A-BA3A-DF11-BA4C-0017A477040C.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/3CBA7F7C-BA3A-DF11-9ECE-0017A4770C14.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/443CAD79-BA3A-DF11-9F90-0017A4770818.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/4C91A47A-BA3A-DF11-B3D2-0017A4771004.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/5225C429-BB3A-DF11-AD90-0017A4770020.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/62BC7102-BB3A-DF11-8D7C-0017A4771028.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/64FCA77B-BA3A-DF11-8514-0017A477042C.root'
     ,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/7AE57478-BA3A-DF11-BA3C-0017A4771034.root'
    )
)

## load jet corrections
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.prefer("ak5CaloL2L3")

#process.content = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(#process.content *
                     process.topDiLeptonOfflineDQM +
                     process.topSingleLeptonDQM +
                     process.dqmSaver
                     )

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

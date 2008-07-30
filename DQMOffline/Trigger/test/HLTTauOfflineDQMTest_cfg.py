import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )


process.source = cms.Source("PoolSource",
               fileNames = cms.untracked.vstring(
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/0E5A2E80-BD42-DD11-A3CE-001617E30F48.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/101F2AF7-B542-DD11-91A2-001617DBCF90.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/14AD2056-BF42-DD11-B44D-001617DBD230.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/14C8BA0E-C042-DD11-973D-000423D6CA42.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/1C88C575-CC42-DD11-80BA-000423D992A4.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/2A13C638-B642-DD11-9E37-000423D991F0.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/301B7700-BC42-DD11-BEE3-001617C3B710.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/347C496A-B642-DD11-B3C3-000423D99F1E.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/388F5BA6-BF42-DD11-BCEF-001617DBD5B2.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/3E163D4D-BE42-DD11-A53B-000423D985E4.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/40546EF7-B642-DD11-84BF-001617E30D52.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/445F37B8-BE42-DD11-B0C1-001617E30E28.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/52498C36-BC42-DD11-9BF8-001617E30F4C.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/56C1088D-BD42-DD11-B2EC-001617E30F58.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/5AD3C30E-B642-DD11-9443-001617C3B79A.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/64A3AD31-BB42-DD11-900B-000423D9853C.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/70E58876-BB42-DD11-8F6E-001617C3B706.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/7259BA57-BF42-DD11-AB71-000423D9853C.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/72D6E3AB-B542-DD11-836C-001617E30CA4.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/7C7DCD69-BC42-DD11-9AEC-000423D6B42C.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/7E27B559-B642-DD11-9C4B-001617DBCF90.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/7E5085A1-C042-DD11-8DA6-001617C3B6E8.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/8A2A7D71-C242-DD11-9DA4-000423D94700.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/9295E136-B742-DD11-9425-001617DF785A.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/9A22E912-B742-DD11-9DF8-000423D8FA38.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/A84B2244-C042-DD11-AE21-000423D9863C.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/AA43B0CB-B542-DD11-81C0-000423D992DC.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/AC6B8E49-BC42-DD11-AF90-001617E30CC8.root',
                        '/store/relval/2008/6/25/RelVal-RelValQQH1352T-1214239099-STARTUP_V1-2nd/0007/AC9E18FF-BC42-DD11-8413-000423D992A4.root',
                         )
               )


process.load("FWCore.MessageService.MessageLogger_cfi")
#process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Load Tau offline DQM
process.load("DQMOffline.Trigger.Tau.HLTTauDQMOffline_cff")

process.DQM.collectorHost = "pcwiscms10"
process.DQM.collectorPort = 9091
process.dqmEnv.subSystemFolder = "HLTOffline/HLTTAU"

process.p = cms.Path(process.dqmEnv)


import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(2000)
        )


process.source = cms.Source("PoolSource",
               fileNames = cms.untracked.vstring(
        '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FED570E8-D6C5-DE11-9C56-001D09F244DE.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FEB11219-F1C5-DE11-BFA6-000423D951D4.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE5A08A3-C3C5-DE11-9355-001D09F2AD84.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE4EE8DE-EAC5-DE11-9AC6-000423D174FE.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE30D14E-CEC5-DE11-81C2-001D09F2426D.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE139931-AFC5-DE11-82CD-001D09F27067.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE0E2554-D3C5-DE11-A511-000423D94908.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE011B69-FDC5-DE11-AC0E-0019B9F70468.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FCFF3131-AFC5-DE11-905D-001D09F2462D.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FC8C5709-C5C5-DE11-9232-001D09F2525D.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FAFF41C7-FBC5-DE11-816F-001D09F244DE.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FA653487-F2C5-DE11-B53C-003048D2BE08.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FA4E8F1E-D6C5-DE11-BAD4-001617E30CD4.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FA1AAC29-E7C5-DE11-8CFD-001D09F295FB.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F82A5AAD-E8C5-DE11-884B-001D09F232B9.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F80FC71B-F1C5-DE11-A916-001D09F2A49C.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F6C77F24-C5C5-DE11-B890-001617C3B6C6.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F615E817-A8C5-DE11-A596-000423D9853C.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4F9D5BE-E5C5-DE11-926E-001D09F24F65.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4D8043C-E2C5-DE11-A23F-001D09F24498.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4BD0F56-D3C5-DE11-A818-003048D3756A.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F45A3B3D-E2C5-DE11-9B72-001D09F295A1.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F43D9F49-E4C5-DE11-A5BA-003048D2C0F0.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F41B91EC-FDC5-DE11-9219-001D09F297EF.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4046603-CAC5-DE11-ADFC-000423D9853C.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F2FFE4D2-BEC5-DE11-99D7-001D09F25456.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F2F97D74-CBC5-DE11-9FA9-000423D9863C.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F0FED93D-A8C5-DE11-A7EA-000423D6CA72.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F09186EA-B6C5-DE11-992F-000423D98B08.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F0889692-E4C5-DE11-809C-000423D98EC4.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F054CDA5-ABC5-DE11-A27A-0019B9F70468.root',
                '/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F0213441-FFC5-DE11-85A4-001D09F253C0.root'
                         )
                            )



process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")


#process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Load Tau offline DQM
process.load("DQMOffline.Trigger.HLTTauDQMOffline_cff")

process.DQM.collectorHost = "pcwiscms10"
process.DQM.collectorPort = 9191
process.dqmEnv.subSystemFolder = "HLTOffline/HLTTAU"

#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/B/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)


process.p = cms.Path(process.HLTTauDQMOffline*process.dqmEnv)

process.o = cms.EndPath(process.HLTTauDQMOfflineHarvesting*process.HLTTauDQMOfflineQuality*process.dqmSaver)

process.schedule = cms.Schedule(process.p,process.o)

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")


process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQMOffline.Trigger.DQMOffline_Trigger_cosmics_cff")
process.load("DQMOffline.Trigger.DQMOffline_Trigger_Client_cff")
process.load("DQMOffline.Trigger.DQMOffline_HLT_Client_cff")




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
        cms.untracked.vstring(
'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FED570E8-D6C5-DE11-9C56-001D09F244DE.root',
'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FEB11219-F1C5-DE11-BFA6-000423D951D4.root'
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE5A08A3-C3C5-DE11-9355-001D09F2AD84.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE4EE8DE-EAC5-DE11-9AC6-000423D174FE.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE30D14E-CEC5-DE11-81C2-001D09F2426D.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE139931-AFC5-DE11-82CD-001D09F27067.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE0E2554-D3C5-DE11-A511-000423D94908.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FE011B69-FDC5-DE11-AC0E-0019B9F70468.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FCFF3131-AFC5-DE11-905D-001D09F2462D.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FC8C5709-C5C5-DE11-9232-001D09F2525D.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FAFF41C7-FBC5-DE11-816F-001D09F244DE.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FA653487-F2C5-DE11-B53C-003048D2BE08.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FA4E8F1E-D6C5-DE11-BAD4-001617E30CD4.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/FA1AAC29-E7C5-DE11-8CFD-001D09F295FB.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F82A5AAD-E8C5-DE11-884B-001D09F232B9.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F80FC71B-F1C5-DE11-A916-001D09F2A49C.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F6C77F24-C5C5-DE11-B890-001617C3B6C6.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F615E817-A8C5-DE11-A596-000423D9853C.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4F9D5BE-E5C5-DE11-926E-001D09F24F65.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4D8043C-E2C5-DE11-A23F-001D09F24498.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4BD0F56-D3C5-DE11-A818-003048D3756A.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F45A3B3D-E2C5-DE11-9B72-001D09F295A1.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F43D9F49-E4C5-DE11-A5BA-003048D2C0F0.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F41B91EC-FDC5-DE11-9219-001D09F297EF.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F4046603-CAC5-DE11-ADFC-000423D9853C.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F2FFE4D2-BEC5-DE11-99D7-001D09F25456.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F2F97D74-CBC5-DE11-9FA9-000423D9863C.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F0FED93D-A8C5-DE11-A7EA-000423D6CA72.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F09186EA-B6C5-DE11-992F-000423D98B08.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F0889692-E4C5-DE11-809C-000423D98EC4.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F054CDA5-ABC5-DE11-A27A-0019B9F70468.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/F0213441-FFC5-DE11-85A4-001D09F253C0.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/EEDDB17B-D8C5-DE11-B330-001D09F25208.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/ECF4FE62-DAC5-DE11-971B-0019DB29C614.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/ECE20281-FCC5-DE11-B90D-000423D99660.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/ECA44857-C9C5-DE11-97CB-001D09F24489.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/EAF6587F-E6C5-DE11-A8D6-000423D951D4.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/EA1005D3-DBC5-DE11-96A4-001D09F282F5.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E8BB205B-E4C5-DE11-A463-001D09F2841C.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E89202EF-DDC5-DE11-8863-003048D37456.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E8544773-CBC5-DE11-802B-001D09F29321.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E4D79902-E8C5-DE11-9B15-001D09F2516D.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E4940531-AFC5-DE11-9AF6-001D09F25217.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E43D3CC7-E3C5-DE11-A88A-001617E30F50.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E2F9FD42-E2C5-DE11-810D-001D09F29619.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E2DF8A57-C4C5-DE11-B602-000423D9880C.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E2DA7B38-CAC5-DE11-9A88-001617E30D40.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E2CDD41F-F1C5-DE11-8568-001D09F29146.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E2666EE7-E2C5-DE11-8C5A-001D09F24D8A.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E25F20F4-F8C5-DE11-93F0-001D09F244BB.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E22E1976-EBC5-DE11-9A0E-001D09F2A465.root',
#'/store/express/Commissioning09/OfflineMonitor/FEVTHLTALL/v8/000/118/969/E20F4C93-B0C5-DE11-BD79-000423D992A4.root'

				)

                            
)


process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.load("DQMServices.Components.DQMStoreStats_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo= cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(100000)
      )
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('dqmHLTFiltersDQMonitor'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
#    destinations = cms.untracked.vstring( 'critical', 'cout')
)

# offline DQM
#process.load("DQMOffline.Configuration.DQMOfflineCosmics_cff")
#process.load("DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff")

process.triggerCosmicOfflineDQMSource.remove(process.l1tmonitor)

#Paths
#process.allPath = cms.Path( process.DQMOfflineCosmics * process.DQMOfflineCosmics_SecondStep)
process.allPath = cms.Path( process.triggerCosmicOfflineDQMSource * process.triggerOfflineDQMClient * process.hltOfflineDQMClient * process.dqmStoreStats )
#process.allPath = cms.Path( process.triggerCosmicOfflineDQMSource*process.hltOfflineDQMClient)
#process.allPath = cms.Path( process.DQMOfflineCosmics)
#process.psource = cms.Path(process.dqmHLTFiltersDQMonitor)

process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Offline'
#process.dqmSaver.workflow = '/Cosmics/CMSSW_2_2_X-Testing/RECO'
process.dqmSaver.workflow = '/StreamHLTMON/Commissioning09-Express-v8/DQMOffline-Trigger-06-02-09-DQM-HLTEvF-05-00-20-2Files'
#process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = True



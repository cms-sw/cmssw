import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

### Miscellanous ###

# Logging #
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool( True )
)
process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'INFO' )
    )
)

# # Profiling #
# process.ProfilerService = cms.Service( "ProfilerService",
#     paths = cms.untracked.vstring(
#         'FullEvent'
#     )
# )

# Memory check #
process.SimpleMemoryCheck = cms.Service( "SimpleMemoryCheck",
#     oncePerEventMode = cms.untracked.bool( True ),
    ignoreTotal      = cms.untracked.int32( 0 )
)

### Import ###

# Magnetic fiels #
# process.load( "Configuration.StandardSequences.MagneticField_0T_cff" )
process.load( "Configuration.StandardSequences.MagneticField_38T_cff" )
# Geometry #
process.load( "Configuration.StandardSequences.GeometryRecoDB_cff" )
# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
# process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'CRAFT_30X::All'
process.es_prefer_GlobalTag = cms.ESPrefer(
    'PoolDBESSource',
    'GlobalTag'
)

### SiStrip DQM ###

process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

### Input ###

# Source #
process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        # run 70036 (B=3.8T)
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0001/F080CBFA-B2F9-DD11-9558-001A92810ACE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0002/14FAA10E-6BFB-DD11-BFE1-003048767DFB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/0042213F-92FA-DD11-ABB0-001731A2832F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/00CA6A13-52F9-DD11-9001-00304875AC63.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/082914F3-B0FA-DD11-B092-001A92971B8E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/0E32EB98-5AFC-DD11-99E0-0018F3D09682.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/2289EA6A-98FA-DD11-A6DE-001731AF6653.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/307E4631-5DFC-DD11-894A-003048767EE3.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/3AA59DEB-7CF9-DD11-8616-003048678AC0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/40F52299-5AFC-DD11-B812-001A92971B48.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/4899CA56-5DFC-DD11-B4D0-001A92811726.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/5E8A0860-B1FA-DD11-8F56-0018F3C3E3A6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/64B1EA8F-EBF9-DD11-99E8-003048678BAA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/78211598-5AFC-DD11-999C-0018F3D09682.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/7EABFC1E-5CFC-DD11-B457-001A92810AC6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/92EEFCF7-4BFA-DD11-8ADD-003048678BE6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/A24F9C9F-62F9-DD11-AE47-0018F3D0969A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/A2793BB2-5BFC-DD11-AAC7-001A928116CC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/AA7FB73B-9C02-DE11-9AFB-003048679084.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/B0734C54-B1FA-DD11-90D7-001A92971AA4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/B48A7D5E-01FA-DD11-966D-001A92811702.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/B6C530E7-51F9-DD11-A47E-00304867BEDE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/C842701E-5DFC-DD11-BADA-001A92810A96.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/CEA9A6A2-46FA-DD11-B265-003048678B30.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/DA17DB5B-ACFA-DD11-AF50-0018F3D0966C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/E0765798-5AFC-DD11-9FFF-0018F3D096E8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/E44AF2F4-D8F9-DD11-87A2-001A928116EA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0007/ECDA429F-73F9-DD11-A983-003048678FAE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/0620D7C7-15FA-DD11-A1B1-001A928116FC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/06C8648C-62F9-DD11-BCA9-003048678C9A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/1465D2F1-6AFA-DD11-B6F7-0018F3D09660.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/2067198B-9C02-DE11-B094-003048754E55.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/20CA6B70-B0F9-DD11-8690-001A92811724.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/304FE1A6-C4F9-DD11-B0C5-001A928116CC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/346569B1-DFFA-DD11-A5C6-0018F3D0969C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/3C83EEF8-A7F9-DD11-BBA7-001A9281174C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/3EFF287C-C6F9-DD11-87C5-003048678A72.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/4E5E8665-BFFA-DD11-BFF5-001731A2832F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/52FDCFC8-9CFA-DD11-9DAC-0018F3D0962C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/66E1F2D3-4FF9-DD11-BC00-0018F3D09706.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/6CCC478B-47FA-DD11-A63B-001A92971AAA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/728ED745-60FC-DD11-A62C-0018F3D09700.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/767FE578-9C02-DE11-A1B3-001731A28799.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/8C2BF633-60FC-DD11-97BC-0018F3D09702.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/A044DE1E-95F9-DD11-8A6B-001A928116B0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/A4956EBD-A7F9-DD11-9506-003048679000.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/AA53E82A-79FA-DD11-A3F5-003048678B5E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/AAB14149-8AFA-DD11-8008-001A92971AD8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/B020218A-D7F9-DD11-ABA5-003048678BC6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/BA13C038-F1F9-DD11-A57C-003048678FF4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/BEF45D1E-74FA-DD11-9961-0018F3D096D8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/C2E61066-A6FA-DD11-A62C-0018F3D0960A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/C4BD031D-7FF9-DD11-B365-003048D15DCA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/C8D8A12B-60FC-DD11-8D7A-0018F3D0968A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/D2C063E3-5AFA-DD11-8590-001A92810AD2.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/D8065E8A-9C02-DE11-AA19-00304876E561.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0009/F4C703FB-4CFA-DD11-B03E-0030486790A6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0011/143573E8-47FC-DD11-A22B-001731AF692D.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0011/60B2B0C3-9B02-DE11-BCFB-001A92971AAA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0011/DA526FF0-ACFA-DD11-A4AD-0018F3D095F6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0011/F8E8D7BA-B1FA-DD11-BB43-0018F3D0960A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/00E5B8ED-B8F9-DD11-AC7A-001A92971B0C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/04EA5F84-ADFA-DD11-82EB-0018F3D096AE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/0C983DB3-E6F9-DD11-9C36-003048754FBB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/129607B5-99F9-DD11-96B3-003048678D6C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/189FA44B-2EFA-DD11-87EF-0018F3D09644.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/2A011F8A-4AFC-DD11-BA09-0018F3D0962E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/2A3B2098-FEF9-DD11-A594-0018F3D09698.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/2CBBB784-34FA-DD11-83A0-003048678E6E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/3AFAFB79-90F9-DD11-BB14-003048678B00.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/440CFF63-A0F9-DD11-ACA3-003048678F92.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/500EA741-2AFA-DD11-B045-001A92811700.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/523E453C-50FA-DD11-90B3-0018F3D096A6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/52B9E258-8AF9-DD11-8544-001A92810AEC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/5EADAB42-E6F9-DD11-AE39-00304876A061.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/5ED60E40-D5F9-DD11-A9E4-003048679182.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/84C6EDB6-77FA-DD11-BC47-003048678FE0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/88D5658E-0BFA-DD11-9D45-003048D15E14.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/94D22E30-93F9-DD11-A5B2-001A92810ADE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/A29A9AC9-E4F9-DD11-AEAA-00304876A075.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/CA53E398-E1F9-DD11-BCCF-001A92810AB6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/CA558D86-CFF9-DD11-9FB8-003048678D6C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/CE7D8DDD-5AFA-DD11-88B6-003048D15CC0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/D277D862-BEFA-DD11-9A49-001731AF68C1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/E218F317-E4F9-DD11-AC3F-0018F3D0965C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/E68B861C-C6F9-DD11-B0A2-0018F3C3E3A6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/F0777209-7FF9-DD11-A6BB-003048678D6C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/F4E08DBA-B7FA-DD11-AC8B-001A9281173C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0012/FEC7F1F4-5FFA-DD11-BB33-003048678B74.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/065CA39A-9FF9-DD11-887D-0030486792B8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/1A6ECFEE-DFF9-DD11-8203-0018F3D09636.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/20FB7EDB-7EF9-DD11-AA60-003048D15E24.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/3645276C-6CFA-DD11-BBD8-001731AF66F7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/36BC899A-51F9-DD11-B48F-001A92971B8C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/42736A84-A0F9-DD11-B690-001731AF68C7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/481DBD2D-9FF9-DD11-A0F9-003048678FFA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/607CAB41-4BFC-DD11-9EFE-0018F3D09648.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/60973C40-4BFC-DD11-A989-001A9281170C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/64E39A83-29FA-DD11-915F-003048678FE4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/6A20644E-4BFC-DD11-9E17-001A92810AEC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/6E9143F9-94F9-DD11-AF21-001A92810A96.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/7E2C5D5E-70F9-DD11-B37B-00304875AAF7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/8280C841-F1F9-DD11-9B29-00304875AA4F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/947B6040-4BFC-DD11-A104-0018F3D09600.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/9C5ADD9C-51F9-DD11-9537-001A92971B80.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/A4BBAC20-D1F9-DD11-9208-003048679296.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/AA4F866C-4BFC-DD11-8617-0018F3D09688.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/ACA8BA41-4BFC-DD11-A8C3-0018F3D095FC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/B0A4C30E-DCF9-DD11-9E2E-001731AF6849.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/B2370EBD-EEF9-DD11-9B59-003048678B06.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/C47DDA5D-81FC-DD11-B83D-001BFCDBD154.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/CA614B86-A1F9-DD11-BFE1-00304867D838.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/D41E1F80-4BFC-DD11-8717-0018F3D09688.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/E89CD83E-4BFC-DD11-AFE2-001A9281172E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/F0026FF9-6BF9-DD11-95D5-001731AF6997.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/F8817813-25FA-DD11-A468-0018F3D09710.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0013/FA897016-E4F9-DD11-B9DE-001A928116B4.root'
#         # run 70421 (B=0T)
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0001/A2C7E9B1-ADF9-DD11-8CD2-003048769E63.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0002/307840F4-6AFB-DD11-9C13-003048767E5D.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/004C5CAC-9B02-DE11-B1F0-001BFCDBD1BE.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/0677A72F-D2F9-DD11-AE7D-0018F3D09600.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/0C047DEE-E6F9-DD11-BE02-0017312B5DA9.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/1277EDDC-7CFA-DD11-87E4-003048678F62.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/12B95EA6-BFF9-DD11-A2DF-003048678AFA.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/1C4F2CCA-79F9-DD11-A1A9-00304867C1BC.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/2008F00D-78FA-DD11-B9A5-003048767EF7.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/22EAC1C7-17FA-DD11-8932-003048678B06.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/38676BED-44FC-DD11-9C95-003048769D5F.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/3AC4756E-16FA-DD11-A8FD-001731AF6651.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/48AFC9E6-44FC-DD11-8A19-003048767E51.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/54519CE6-76F9-DD11-BB03-00304867918E.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/5689B62A-74FA-DD11-B070-0018F3D0968A.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/5C97C70F-45FC-DD11-A3A6-001A92810AB8.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/62A504FB-44FC-DD11-A9FF-00304876A0DB.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/646E1BF1-44FC-DD11-9651-00304876A15B.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/649E04E2-44FC-DD11-827E-00304876A147.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/686530D5-70FA-DD11-868B-0018F3D09648.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/8477924A-BCF9-DD11-9268-00304867916E.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/8E416018-C5F9-DD11-82CE-003048679070.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/9AC6AFBF-83F9-DD11-B0EB-001A928116F0.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/B0E22FA6-95F9-DD11-9950-003048754FB9.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/B456BFF1-44FC-DD11-939A-001731A28F19.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/CC5755DE-5BF9-DD11-9EDC-0018F3D096F0.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/DE8C76B0-93FA-DD11-A936-003048767DDB.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/E2E91354-E8F9-DD11-8E5A-001731A28998.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/E41B0CA5-45FC-DD11-B5B8-001731AF66AF.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/E8FB10F9-44FC-DD11-9E9D-00304875AA6F.root',
#         '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/F4549C4F-96FA-DD11-A79C-001731AF6BBF.root'
#         # run 70036 (B=3.8T), TrackerPointing with corr. cabling
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/0648D214-7B0B-DE11-A940-001A92811718.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/06DD7C0C-800B-DE11-BF8E-0018F3D09650.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/2EF8C7AF-7B0B-DE11-A20F-003048678ADA.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/3C967497-7C0B-DE11-A361-00304867905A.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/5E14F4C9-7F0B-DE11-A2B3-001731EF61B4.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/7C67DE9A-7B0B-DE11-8DF1-001A9281173C.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/8293D24F-780B-DE11-B3A9-0018F3D09608.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/8CDCE5C9-7E0B-DE11-ACA8-00304867C0C4.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/94D7DC48-7E0B-DE11-A88E-00304867C0F6.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/B0F2785A-760B-DE11-9584-001731AF65E5.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0009/C0320B6E-7B0B-DE11-9E62-00304867C026.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/0E7F922C-8C0B-DE11-A105-001A9281174C.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/16BE6970-840B-DE11-A9D5-003048678ED4.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/52AE464D-8F0B-DE11-A30A-003048678B7E.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/72A0DBB1-8D0B-DE11-BBBF-003048678BC6.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/AC06D4AD-8A0B-DE11-99C4-003048678B94.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/B2FC9935-8C0B-DE11-B6CA-001A92971B38.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/BCF0B404-860B-DE11-AF56-0017312B58FF.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/D0762288-870B-DE11-A89B-003048767FB3.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/DE075BCD-8D0B-DE11-9BBB-003048678CA2.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/E6A5FB4D-8F0B-DE11-B071-00304867C026.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0010/EA59323E-830B-DE11-8605-001731AF65CF.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0011/1A8B7DF2-900B-DE11-A104-00304876A0D7.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0011/2AE67FA8-9F0B-DE11-8DD2-001A92971B0C.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0011/AC524810-9A0B-DE11-A25B-003048678B84.root',
#         '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_225_ReReco_FromTrackerPointing_v1/0012/C8AACB2F-280C-DE11-9CFB-003048679182.root'
    )
)
# Input steering #
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10000 )
)

# HLT Filter #
process.hltFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths           = cms.vstring(
#         'HLT_WhatEverFunnyFilter',
#         'HLT_TrackerCosmics',
#         'HLT_TrackerCosmics_CoTF',
#         'HLT_TrackerCosmics_RS'  ,
        'HLT_TrackerCosmics_CTF'
    ),
    eventSetupPathsKey = cms.string( '' ),
    andOr              = cms.bool( True ),
    # use this according to https://hypernews.cern.ch/HyperNews/CMS/get/global-runs/537.html
    TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'HLT' )
#     TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'FU' )    
)

### Scheduling ###

process.p = cms.Path(
#     process.hltFilter                            * # comment this out to switch off the HLT pre-selection
#     process.SiStripDQMRecoFromRaw                * # comment this out when running from RECO or with full reconstruction
#     process.SiStripDQMSourceGlobalRunCAF_fromRAW * # comment this out when running from RECO or with full reconstruction
    process.SiStripDQMRecoGlobalRunCAF           *
#     process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripMonitorClusterCAF             *
#     process.SiStripOfflineDQMClient              *
#     process.qTester                              *
    process.dqmSaver
#     process.MEtoEDMConverter
)

### Output ###

# DQM Saver path
process.dqmSaver.dirName = '.'

# PoolOutput #
process.out = cms.OutputModule( "PoolOutputModule",
    fileName       = cms.untracked.string( './SiStripDQMOfflineGlobalRunCAF.root' ),
    SelectEvents   = cms.untracked.PSet(
        SelectEvents = cms.vstring( 'p' )
    ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_SiStripDQMOfflineGlobalRunCAF'
    )
)

# process.outpath = cms.EndPath(
#     process.out
# )

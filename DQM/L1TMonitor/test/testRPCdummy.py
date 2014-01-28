import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")
#process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('/tmp/log.txt')
)


process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.test.dqm_onlineEnv_cfi")

process.load("L1Trigger.HardwareValidation.L1DummyProducer_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cff")

process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCPatSource_cfi")
#process.rpcconf.filedir = cms.untracked.string('RPCData/L1RPCData/data/CosmicPats6/')
#process.rpcconf.PACsPerTower = cms.untracked.int32(1)

process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff")
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomConfig_cff")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeSource_cfi")

process.load("L1Trigger.RPCTrigger.rpcTriggerDigis_cfi")

process.load("DQM.L1TMonitorClient.L1TRPCTFClient_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

            '/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/002005DE-87AE-DD11-BFED-001EC9AA99A5.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/006621D8-80AE-DD11-9658-00E081339578.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/025E9843-8BAE-DD11-8046-001EC9AA9FE5.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/025F37B3-7BAE-DD11-A8D8-0030483345FC.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/02CEDAFD-87AE-DD11-AA4D-0030483345E4.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/02D8C75F-84AE-DD11-9D44-003048359D9C.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/041B864B-95AE-DD11-8AFF-001E682F1F20.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0467F404-81AE-DD11-BF8E-00E08123791F.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/048A6975-CDAE-DD11-BE24-001A9227D3A9.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/04B86398-82AE-DD11-8C3A-001EC9AAB911.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/04C3760D-86AE-DD11-8C36-0019B9CAC0F8.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/04C4E884-7DAE-DD11-B726-00E0813269C4.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0601B306-7FAE-DD11-BCBA-00E08133CD36.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0618AC5F-84AE-DD11-9AB9-001EC94BA146.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0650C7FE-87AE-DD11-8F66-00304832293E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0656DCCC-99AE-DD11-AC8F-001E8CC04112.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0660CD41-7FAE-DD11-A659-00304832293E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0670FDFD-87AE-DD11-9C12-0030487D07B6.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/06B7DE54-CCAE-DD11-8758-001A92544626.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/06E9B6DC-80AE-DD11-A97A-00E08133D4A2.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0826F060-84AE-DD11-BAD2-003048359D9C.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/083C4BDD-87AE-DD11-B1E0-00E08133F12A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/087F13D7-87AE-DD11-BD82-001EC9AA9EA5.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/088921D8-87AE-DD11-AE54-001EC9AAD5C6.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/08B98B5C-78AE-DD11-A2A1-00304834BB58.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/08D807E6-80AE-DD11-BBA8-001EC94BF71A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/08E01840-8BAE-DD11-8046-001EC9AAD68E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/08E43445-84AE-DD11-BE53-0019B9CAFE71.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0A0235A4-82AE-DD11-80CF-001EC9AAD611.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0AA16C55-7DAE-DD11-9E68-003048322BD2.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0AAD6C07-92AE-DD11-AA65-001E682F1F94.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0C0AF4BB-A1AE-DD11-9536-003048770DCE.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0C530189-BAAE-DD11-B2E7-003048770C5A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0CB30507-F6AE-DD11-985A-00304858A6A7.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/0EC8A39D-82AE-DD11-9882-0016368E0DE0.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/103B0BC3-7BAE-DD11-8FCB-00E08133F12A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/105AD10F-86AE-DD11-8E68-00E081339574.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1092C841-7FAE-DD11-80C1-003048335548.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/10A50E54-78AE-DD11-B955-001EC94BFDD9.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/10AD1F5E-78AE-DD11-AE47-001EC94BE9F4.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/12C68FB1-7BAE-DD11-85C0-003048553D02.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1431C8D7-87AE-DD11-9C06-00E08134B780.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/145E6D56-7DAE-DD11-9980-00304833557A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1463D305-9AAE-DD11-9D9A-00304858A6A7.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/14EF6ED4-80AE-DD11-B22F-00E08133E4B2.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/164B0008-9AAE-DD11-980A-001E8CCCE148.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/168DB940-7FAE-DD11-9664-00304834BB58.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/16CBF656-84AE-DD11-B87B-00E081333FAE.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/16EF1C55-7DAE-DD11-BDAF-001EC9AA9DBA.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/185EE3B5-7BAE-DD11-B5E7-001C23C0F175.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1886518D-C2AE-DD11-AB60-001A9243D52E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/18F363AA-7BAE-DD11-86A0-00E08143CF45.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1A6E6E56-84AE-DD11-B889-001C23BED6CA.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1C1B6CDD-87AE-DD11-8214-001EC9AAA288.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1C29F756-84AE-DD11-AD59-00E081326D18.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1CC56909-B4AE-DD11-B02D-001E681E0F9A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1CD6AC4C-95AE-DD11-9152-001E681E100E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1E230752-78AE-DD11-900E-001EC94B51EE.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1E7D3356-84AE-DD11-9CED-0030487D07BA.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1E7D971D-86AE-DD11-823D-001EC9AAA32D.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/1E879567-84AE-DD11-9215-00E081429664.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/209E6882-7DAE-DD11-A3E3-00E08134C25E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/20CC853F-84AE-DD11-8987-0019B9CADC3D.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/20DA42A1-89AE-DD11-9024-00E0813289DE.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2214CD3A-CFAE-DD11-AE99-001A925444DC.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/221DAB5A-7DAE-DD11-9716-00163691DF32.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/22351D1D-86AE-DD11-8593-001EC9AA9BD0.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/224C7CDB-A1AE-DD11-950B-003048770CD0.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/225D109A-89AE-DD11-B425-001EC9AA9F90.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2264B423-78AE-DD11-8109-00304865C478.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/22A1D441-7FAE-DD11-83E0-003048359D9C.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/22CBC9C7-8EAE-DD11-850E-0030482E9BC2.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/242257C3-E8AE-DD11-A108-001A9254453A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/243DD750-84AE-DD11-B743-0030482CD990.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/249ACD6B-84AE-DD11-89FA-0019B9CACEF7.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/24BB5031-7FAE-DD11-8A0E-00163691DA92.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/260D5806-7CAE-DD11-87F3-001E8CCCE114.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/26207F1D-85AE-DD11-A612-003048770C30.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/262F3A19-8DAE-DD11-93C5-00163691D99E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/264A8820-86AE-DD11-9D9F-001EC9AA9EFA.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/264B56C8-96AE-DD11-9D0A-001A9243D528.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2655A941-7FAE-DD11-A674-0030483344E2.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/26989242-7FAE-DD11-9C5F-00304833457E.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/269B99D7-80AE-DD11-A284-00E0813267BC.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/26B683A8-82AE-DD11-A37C-003048553B88.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/26CD5B1F-86AE-DD11-97B6-001EC9AAD5C6.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/284363E9-87AE-DD11-BA85-001EC9AA9158.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/28462DC8-B8AE-DD11-BDCA-001A9227D3D1.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2858BF12-7FAE-DD11-A7AB-001EC9AAD5F3.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/289D2301-88AE-DD11-8D5C-00E0812E9E16.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/28BC6137-A4AE-DD11-8334-001E682F1F9A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2A0B322A-81AE-DD11-8CDD-001EC9AAA2EC.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2A2AAD0D-86AE-DD11-A19D-00E08133CDA0.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2A30C79F-D8AE-DD11-9B88-0002B3D90F13.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2A6F74E1-87AE-DD11-AA15-00E08142962A.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2AA1C745-9EAE-DD11-A9F8-003048770C64.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2AA31830-7FAE-DD11-A684-00E08134420C.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2AE2D247-78AE-DD11-B522-003048553D1C.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2C278175-9BAE-DD11-A8B9-001E8CCCE140.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2C28BB66-83AE-DD11-A4F1-0030482CC1AE.root'
            ,'/store/mc/Summer08/CosmicMCBOn10GeV/RECO/COSMMC_21X_v4/0004/2C37EC0F-86AE-DD11-A7BD-00163691D18E.root'

    )
)

process.l1trpctf = cms.EDFilter("L1TRPCTF",
    rpctfRPCDigiSource = cms.InputTag("muonRPCDigis"),
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False),
    rpctfSource = cms.InputTag("gmtDigis"),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        l1dummy = cms.untracked.uint32(1762349)
    )
)

process.p = cms.Path(process.rpcTriggerDigis*process.l1dummy*process.gmtDigis*process.l1trpctf*process.l1trpctfqTester*process.l1trpctfClient*process.dqmEnv*process.dqmSaver)
process.MessageLogger.destinations = ['log.txt']
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1T'
process.l1dummy.VerboseFlag = 0
process.l1dummy.DO_SYSTEM = [0, 0, 1, 0, 0, 
    1, 0, 1, 0, 0, 
    0, 0]
process.gmtDigis.DTCandidates = 'l1dummy:DT'
process.gmtDigis.CSCCandidates = 'l1dummy:CSC'
process.gmtDigis.MipIsoData = 'l1dummy'
process.rpcTriggerDigis.label = 'muonRPCDigis'
process.l1trpctfClient.verbose = True



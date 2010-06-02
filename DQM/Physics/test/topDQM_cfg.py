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
     # '/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/1A19B479-BA3A-DF11-8E43-0017A4770410.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/1A2CED78-BA3A-DF11-98CD-0017A4771010.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/3AE61B7A-BA3A-DF11-BA4C-0017A477040C.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/3CBA7F7C-BA3A-DF11-9ECE-0017A4770C14.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/443CAD79-BA3A-DF11-9F90-0017A4770818.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/4C91A47A-BA3A-DF11-B3D2-0017A4771004.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/5225C429-BB3A-DF11-AD90-0017A4770020.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/62BC7102-BB3A-DF11-8D7C-0017A4771028.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/64FCA77B-BA3A-DF11-8514-0017A477042C.root'
     #,'/store/mc/Spring10/TTbar/GEN-SIM-RECO/MC_3XY_V25_S09_preproduction-v2/0106/7AE57478-BA3A-DF11-BA3C-0017A4771034.root'

     '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FEC52E5B-D767-DF11-9A6C-0030486733D8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FEBD6C22-AD67-DF11-8AB1-0030487C6A66.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FE9A8FC7-B267-DF11-A3B1-001D09F24EAC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FE69F9A2-B767-DF11-A1C5-000423D944FC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FE4A23FA-AF67-DF11-9B9C-0030487D05B0.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FE3E1C3E-E367-DF11-8275-001D09F29533.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FE068AA5-DC67-DF11-8706-001D09F29533.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FE0484BA-C567-DF11-B1C4-001D09F2438A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FCA9819F-D667-DF11-BB90-001D09F2423B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FCA92512-C567-DF11-A8FA-000423DD2F34.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FCA83051-BD67-DF11-BE1E-001D09F242EA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FC5016EE-CE67-DF11-9298-0030487A18F2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FC260F3C-A867-DF11-99A1-001D09F25438.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FC19F6F4-C967-DF11-8746-001D09F252DA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FC0507F8-C967-DF11-8755-001D09F2960F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FAF21623-C767-DF11-B262-001D09F28EC1.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FAC6C98C-BD67-DF11-8B2C-001617E30CC8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FAB5C69B-D667-DF11-B4EB-000423D996B4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA9BEAA1-BC67-DF11-98F1-001D09F248F8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA97486D-BF67-DF11-8308-001D09F24D8A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA96C90B-DD67-DF11-AACA-001D09F29533.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA8C9FCB-D367-DF11-AC42-000423D94E1C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA3C313E-E367-DF11-B7EC-0019B9F70607.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA3817F4-DC67-DF11-8286-001D09F232B9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA2E0ABC-A967-DF11-9F62-001D09F2AF1E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA0EC4C6-B967-DF11-A04C-0019B9F4A1D7.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/FA086EA2-BC67-DF11-A4C1-001D09F295FB.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F8F82ECB-DF67-DF11-9019-001D09F2A465.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F8CC30DB-CC67-DF11-B49D-001D09F2527B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F8B59968-BB67-DF11-BB01-001D09F2906A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F862ABA8-B767-DF11-9098-001617DC1F70.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F8585C0E-AB67-DF11-942F-001D09F29146.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F8553A68-C667-DF11-8B41-001D09F28E80.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F83F7381-C167-DF11-A9E3-003048673374.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F83988C7-B267-DF11-8B69-001D09F25456.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F82433B6-AB67-DF11-A39C-001D09F2841C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F6C4816F-C467-DF11-AD7A-001D09F23C73.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F6BA966D-D967-DF11-A020-001D09F24399.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F6B6F66B-BB67-DF11-B248-001D09F2B2CF.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F6A9ADDC-B467-DF11-B558-0030487C912E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F6A8D5B1-A967-DF11-9739-0030487C90EE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F6A72E6B-D967-DF11-867D-000423D944F8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F694D662-B167-DF11-A143-0019B9F704D6.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F681EEDB-A667-DF11-817B-00304879EE3E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F67E859F-D667-DF11-A8FF-001D09F2516D.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F64D6C31-DF67-DF11-8487-000423D6CA02.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F60B4BB0-DC67-DF11-8EB5-000423D6C8E6.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F60378C9-DF67-DF11-A473-0019B9F70607.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F4F711D4-AD67-DF11-8BEB-001D09F24FEC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F4F080AF-B067-DF11-BB69-001D09F2AF96.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F4EA21EA-B767-DF11-A452-001617E30D0A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F4C18896-C367-DF11-A7D8-000423D6BA18.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F4AA0F13-B967-DF11-8E6B-000423D6CA72.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F49C43F5-C967-DF11-BCFB-001D09F2983F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F494C98D-AE67-DF11-A68D-001D09F25041.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F4542DFB-BD67-DF11-821E-001D09F2983F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F423F63F-AF67-DF11-BC5C-0030487C90D4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F40D04B7-D167-DF11-A925-000423D6CA6E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F40673A8-DC67-DF11-98C3-001D09F2516D.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2FA320F-AB67-DF11-A87A-0030487C912E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2D6A831-E167-DF11-B126-001D09F291D7.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2C2428C-B567-DF11-9AA3-001D09F28F25.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2B3F468-C767-DF11-B5A0-001617C3B6E2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2AB718F-AE67-DF11-8BD5-001D09F254CE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2A3E0C3-CC67-DF11-AFA1-0015C5FDE067.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F29B9BF5-C967-DF11-B3DC-001D09F24489.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F27F8FAF-B067-DF11-8F9A-001D09F251D1.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F2588CED-C967-DF11-86B7-001D09F253C0.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F213405B-DE67-DF11-9F4B-001D09F231C9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F20A83DC-B467-DF11-9589-001D09F2527B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F0E1A4F7-B667-DF11-AADB-000423D999CA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F0DB98F2-DC67-DF11-B0FC-000423D6B444.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F0AA52AE-CA67-DF11-9FD7-001D09F28EA3.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F08B80D9-CC67-DF11-AAD4-0030487C90EE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F08279FF-C567-DF11-8CCB-001617C3B710.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F01344F5-C967-DF11-A0CC-001D09F23A3E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/F006BEB5-D867-DF11-BB7F-000423D98DB4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EEE3F763-D967-DF11-B6B3-000423D94E70.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EECFF325-DA67-DF11-9214-001D09F254CE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EEC73933-DF67-DF11-BB4C-000423D992A4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EE970FE2-AD67-DF11-BE46-001D09F2512C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EE7F73ED-CE67-DF11-A314-001D09F2527B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EE6DBCDA-B467-DF11-8A67-0019B9F72CE5.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EE55D615-B967-DF11-8282-000423D6B5C4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EE3B10D4-AD67-DF11-AFC0-001D09F24EE3.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EE013D47-E367-DF11-B17A-001D09F34488.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/ECEC0F23-A667-DF11-B047-000423D98BC4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/ECE75ABB-A967-DF11-840B-001D09F28D4A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/ECDFD957-D767-DF11-91DD-000423D985E4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/ECD62505-D867-DF11-8310-001D09F23D1D.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/ECD2EC80-C167-DF11-8A78-001617E30CC2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC9DFAEE-E367-DF11-A3F5-001D09F282F5.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC7A6F9A-C367-DF11-8DBA-0019DB29C5FC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC7A145F-D767-DF11-A396-003048678110.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC530519-CC67-DF11-A0C9-001D09F253D4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC358189-C667-DF11-BA10-001D09F23C73.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC22E73B-A867-DF11-A020-001D09F295FB.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC1B28FA-AF67-DF11-A758-0030487D0D3A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EC000D66-D967-DF11-8A44-0030487A3DE0.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EAF2A700-BE67-DF11-BC81-001617E30D4A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EAEF419B-C367-DF11-9700-001D09F23944.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EAE9BC0C-AB67-DF11-8A14-0030487C90C2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EADEAB26-DA67-DF11-8E90-001D09F2438A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EAC2F219-DA67-DF11-A13E-001D09F29533.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EABCB145-DC67-DF11-89A9-001D09F28D54.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/EA183F86-C167-DF11-85B0-001D09F24259.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E8E41A7D-D467-DF11-9661-001D09F291D2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E8CD7B89-E267-DF11-9F56-001D09F28F1B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E8983FBB-A967-DF11-9AAA-001D09F27003.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E88D7A03-C367-DF11-9EF7-001D09F24448.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E87DECEF-CE67-DF11-A8E6-001D09F244BB.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E876668C-B567-DF11-A32A-001D09F252F3.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E869B91B-C767-DF11-BFE4-001D09F292D1.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E85D8D8D-B567-DF11-B4A8-001D09F2512C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E830EEC3-DC67-DF11-B1FA-001D09F23944.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E818D508-D867-DF11-B7F2-001D09F2516D.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E81109DC-B467-DF11-9A92-0030487C6090.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E6D977AE-BC67-DF11-ACA9-001D09F23174.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E6C5220B-AB67-DF11-A152-001D09F253FC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E6B8C808-D867-DF11-9835-001D09F23C73.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E6A3E819-DA67-DF11-8BCC-0019B9F730D2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E66F1B21-C067-DF11-8405-003048D2BB90.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E6657BD7-CC67-DF11-AF7C-001D09F251BD.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E62FBB19-CC67-DF11-B820-001D09F29524.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E627C040-D567-DF11-94A6-000423D94700.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E606A97B-CD67-DF11-B4DB-0015C5FDE067.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E4524BC8-B267-DF11-B8AF-001D09F24F1F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E4497225-A667-DF11-A0AB-0030487A18D8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E436D7A7-B867-DF11-BE47-001D09F297EF.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E42A4A1A-DA67-DF11-9BA2-001D09F24DDF.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E4178385-C167-DF11-9070-003048678098.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E2DFB30E-AB67-DF11-9904-0030487D0D3A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E2BCF5D0-A667-DF11-85C7-000423D98834.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E2B7CDF3-C967-DF11-95CC-000423D98DB4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E2B674BD-DC67-DF11-9B23-001D09F29321.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E2A08702-BE67-DF11-916F-001D09F28F25.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E29DD58C-D467-DF11-A996-000423D99AAE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E26CFB3B-D167-DF11-9886-001D09F23C73.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E25ADBA0-C167-DF11-9857-000423D6B444.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E22A9684-C167-DF11-9B80-000423D985E4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E20A3823-D367-DF11-B3CC-0030487C8E02.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E0FC3ACD-C767-DF11-9CA9-001D09F2960F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E0B47A22-AD67-DF11-ACB2-0019B9F707D8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E09B7620-E067-DF11-8AFD-001D09F251B8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E0898E5A-D267-DF11-ABED-00151796CD80.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E06D0769-C667-DF11-99F8-001D09F25208.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E05F0CEC-CE67-DF11-833F-0015C5FDE067.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E05D6E32-CE67-DF11-9287-001D09F28F1B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/E033E3DD-B467-DF11-9FBF-001D09F231B0.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DEF58426-D367-DF11-A5F2-000423D6CA6E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DEE8FD10-B967-DF11-912A-000423D6BA18.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DEE0918D-B567-DF11-BFDE-0019B9F705A3.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DEDA8923-D367-DF11-83C1-001D09F24047.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DEBB0CB5-BA67-DF11-A59B-001D09F292D1.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DEB3DA6B-AC67-DF11-8F4C-001D09F29533.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DE7F832A-CE67-DF11-B26F-001D09F253D4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DE76045B-CB67-DF11-9B5F-001D09F231C9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DE12F6B9-C567-DF11-85C4-001D09F244DE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DE0CF338-DC67-DF11-AD87-001D09F2B2CF.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DCD7E7AC-BE67-DF11-94BB-001D09F25456.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DC2B4A0F-AB67-DF11-B36A-0019B9F581C9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DC160EC7-B267-DF11-A5C5-001D09F28D54.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DAB3A009-C567-DF11-AF52-000423D986A8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DA6630DC-B467-DF11-832E-001D09F29169.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DA4E8323-A667-DF11-9BFA-001D09F24DDF.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/DA17B1DC-B467-DF11-8646-001D09F2905B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8F2D86E-E067-DF11-985F-001D09F251B8.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8E0900E-B967-DF11-9722-000423D9870C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8DB584F-C467-DF11-B078-001D09F23944.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8C92939-D567-DF11-BCE8-003048D37456.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8C2F537-DF67-DF11-9793-000423D98804.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8A8AC3E-AF67-DF11-8545-0030487C8E02.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D855A931-C267-DF11-B388-001D09F2983F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D82A161B-A667-DF11-91DE-0030487A18F2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D8294D63-D767-DF11-A9B3-0019B9F70607.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D81279B6-AB67-DF11-83C7-001D09F244DE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D6CE9D50-C467-DF11-99BF-0019B9F72CE5.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D6BAC9BC-DB67-DF11-9AED-001D09F24259.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D69B2DC8-DB67-DF11-AAC6-0030487CD7EA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D693DFE7-D567-DF11-901E-000423D9A212.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D6925037-DF67-DF11-8BC0-000423D98DB4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D680D10A-C567-DF11-8B6A-000423D9939C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D6689889-E267-DF11-93CE-001D09F24399.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D65DF0EB-C967-DF11-A2FF-001D09F28E80.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D64AE5A7-B867-DF11-BF1A-001D09F28D4A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D4FDC49E-D667-DF11-A828-001D09F231C9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D4EE601C-CC67-DF11-AB68-001D09F27067.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D4CC54E2-BB67-DF11-8389-0030486733B4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D4C14E7F-C167-DF11-987A-003048D2BBF0.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D46FEBAB-DB67-DF11-A71D-0030487A1FEC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2E3AEF7-AF67-DF11-9557-0030487CD7EA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2E08819-DA67-DF11-81F0-001D09F2423B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2D45C23-C767-DF11-B061-001D09F244DE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2D2A06F-AC67-DF11-9E2F-00304879EE3E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2D03851-D067-DF11-8729-003048678110.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2CFCFA2-CF67-DF11-B096-001D09F24FEC.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2C17589-E267-DF11-BBBC-0019B9F72D71.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2B5C6D0-A667-DF11-A9F9-000423D94908.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2AC438A-E267-DF11-9BF8-001D09F29533.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D2921CA1-BC67-DF11-8150-001D09F29321.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D27DF362-BB67-DF11-AAB3-001D09F29597.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D27613DD-B467-DF11-B010-0019B9F72D71.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D25B1746-BD67-DF11-AFF1-001D09F2512C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D24FAB2B-B467-DF11-9287-000423D9517C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D23B49F5-C967-DF11-B88D-001D09F2AD7F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D0BF7CBA-DB67-DF11-8F8B-001D09F25438.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D05DA09B-CF67-DF11-98A5-0030487A18F2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D02B0209-D867-DF11-99DC-0019B9F72F97.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D02749BF-C567-DF11-A3AE-001D09F292D1.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/D0132637-D567-DF11-8ED3-001D09F25041.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CEF2AD8D-B567-DF11-A081-001D09F25456.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CEEEA21C-E067-DF11-8A30-001D09F24FBA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CEC6DE17-CC67-DF11-BF03-0015C5FDE067.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CEBC9EE9-D567-DF11-A1B7-001D09F25041.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CE8D9915-CF67-DF11-9C82-001D09F23C73.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CE7BDC8F-B567-DF11-B2EE-001D09F2983F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CE6F3045-BD67-DF11-BF8E-001617DBD230.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CE459E1C-C067-DF11-BE68-000423D6BA18.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CE44C947-E367-DF11-895A-001D09F25456.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CE410910-CA67-DF11-827B-001D09F2432B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CCF729A4-CA67-DF11-9A01-001D09F232B9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CCD4F280-BB67-DF11-B895-001D09F25393.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CCABF5A6-CA67-DF11-850D-001D09F28F1B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CCABA938-D567-DF11-91C5-003048D2C0F2.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CC93137B-CD67-DF11-B42F-001D09F253D4.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CC6621A7-CA67-DF11-B1FC-0019B9F72F97.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CC6280CD-B267-DF11-8020-001D09F2906A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CC219D5B-CB67-DF11-BB28-001D09F24682.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CC2137F6-DC67-DF11-A2E9-001D09F23D1D.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CC01BFB4-CA67-DF11-9171-001D09F24259.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CAEE42B2-A967-DF11-8B7C-0030487CF41E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CAC52D55-BD67-DF11-96B5-001D09F2924F.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CAB79C7A-CD67-DF11-96B5-001D09F24F65.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CAA5654F-C467-DF11-99A4-001D09F295FB.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CAA3B426-B467-DF11-8C90-000423D999CA.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CA7894CB-B967-DF11-BE52-001D09F23A20.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CA711C53-E367-DF11-81F9-001D09F232B9.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CA56D116-E067-DF11-BF5A-001D09F2906A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CA4B8DDC-B467-DF11-9D7E-001D09F2932B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CA38DE1C-B967-DF11-81A2-000423D6B48C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/CA31CB3D-A867-DF11-B91B-0030487CD13A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C8EB7BF4-C967-DF11-ABD2-000423D99A8E.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C8E85AD2-A667-DF11-8811-0030487A3C9A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C8CD7D6F-D967-DF11-88F5-001D09F254CE.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C8C332F9-CC67-DF11-8BD0-001D09F2432B.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C8AA00D3-C067-DF11-BD39-0019B9F7312C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C898AA23-A667-DF11-9DB4-0030487C608C.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C895DE02-BE67-DF11-8363-001D09F24D8A.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C87C34BB-BE67-DF11-A03A-001D09F24EE3.root',
        '/store/express/Run2010A/ExpressPhysics/FEVT/v2/000/136/100/C86E7DE3-CE67-DF11-BFC4-001D09F24EE3.root'
        


      
    )
)

## load jet corrections
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.prefer("ak5CaloL2L3")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))

#process.content = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(#process.content *
                     process.topSingleMuonMediumDQM +
                     process.topSingleElectronMediumDQM +
                     process.topSingleMuonLooseDQM +
                     process.topSingleElectronLooseDQM +
                     process.topDiLeptonOfflineDQM +
                     process.topSingleLeptonDQM +
                     process.dqmSaver
                     )

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

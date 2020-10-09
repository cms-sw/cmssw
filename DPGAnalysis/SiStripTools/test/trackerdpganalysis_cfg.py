import FWCore.ParameterSet.Config as cms

process = cms.Process("clusterAnalysis")
process.load("FWCore.MessageService.MessageLogger_cfi")



process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
process.MessageLogger.infos.threshold = cms.untracked.string("INFO")
process.MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
        )
process.MessageLogger.infos.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
        )
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")




process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FEF39343-215C-DF11-AD9D-001D09F24399.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FECA7176-FD5B-DF11-984C-0030487A17B8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FE26CE70-2C5C-DF11-B4DC-000423D999CA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FE1759B2-005C-DF11-B86D-001D09F2438A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FCEA6825-0C5C-DF11-9438-000423D95030.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FCE2907F-195C-DF11-B3D4-0030487C8CB6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FCC37DF2-285C-DF11-A8F0-0030487D0D3A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FC6F1257-1A5C-DF11-A867-000423D98FBC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FAC77038-075C-DF11-98A7-003048D2BCA2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/FA05172A-FA5B-DF11-A224-001D09F26509.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F8EEF874-FF5B-DF11-800B-003048673374.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F8EA90EE-F65B-DF11-A494-0030486780AC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F89B2D7E-195C-DF11-A1D3-0030487A3C9A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F8532B18-FC5B-DF11-9E11-0030487CD7C6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F6A6689A-225C-DF11-8FF9-000423D944FC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F684467E-195C-DF11-8E8E-001D09F26509.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F651D4AE-FC5B-DF11-8C36-001617E30CD4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F4F4BD12-FA5B-DF11-93F7-0019B9F72BAA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F4E50941-075C-DF11-9D7F-0030486780AC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F4C16C39-075C-DF11-AF79-001617E30D00.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F4BFC6EE-F95B-DF11-AACD-003048D37538.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F475F1F9-F45B-DF11-ADEF-000423D99264.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F42ACABB-1D5C-DF11-B4B2-000423D6B42C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F2E28466-075C-DF11-A31D-000423D9890C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F255A551-1A5C-DF11-9F77-001D09F29619.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F23CA617-035C-DF11-9FF0-0030486780EC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F2379D94-205C-DF11-A1EE-001617DBD556.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F227E660-2A5C-DF11-875D-0030487C8E00.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/F008B85B-1A5C-DF11-906F-001D09F2438A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/EEDB16F2-215C-DF11-B503-0030487A1990.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/EEBD890B-FA5B-DF11-BA94-0019B9F7312C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/EC369659-235C-DF11-92D5-000423D6C8E6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/EAB8DEF2-0C5C-DF11-B6D1-000423D6BA18.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/EA9CDE88-015C-DF11-8ECE-000423D99EEE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E8FC082B-FA5B-DF11-853E-001D09F24399.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E813E32D-1F5C-DF11-B87D-0030487A17B8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E6CCAB55-F05B-DF11-A24C-001D09F28755.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E6B17445-075C-DF11-951B-003048D2C16E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E65896EE-215C-DF11-99CA-0030487D0D3A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E4F762C5-1D5C-DF11-BA63-000423D98AF0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E4CB4142-075C-DF11-BC7A-000423D94494.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E45A7A70-255C-DF11-B51D-000423D8FA38.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E0EA2A32-1D5C-DF11-A016-000423D996B4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E0B4C8AC-085C-DF11-8BDC-0030487A195C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/E05850C1-1D5C-DF11-ABAF-000423D94908.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DED826BA-165C-DF11-A14F-001617C3B6DC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DE68D4D9-0E5C-DF11-9F3B-000423D991F0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DE3F905A-235C-DF11-BC78-000423D98844.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DE2D13BD-1D5C-DF11-BECB-000423D99AAA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DCE2A696-205C-DF11-9D5B-000423D98750.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DCBDCECB-065C-DF11-AD29-000423D6C8EE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DC80118B-195C-DF11-A49E-001D09F2525D.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DC3A8FEE-005C-DF11-8BB2-000423D99264.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DC344135-1D5C-DF11-B77D-001D09F28F0C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DAE1FC51-1C5C-DF11-8CB9-0030487CD180.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DAB1B491-FA5B-DF11-A879-000423D174FE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/DAA72947-075C-DF11-9AF4-000423D944F8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D8BF38B2-085C-DF11-A649-001D09F244BB.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D87662BC-1D5C-DF11-B9F5-000423D99E46.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D875296A-FF5B-DF11-8450-001617C3B6E2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D849AA52-1C5C-DF11-8898-0030487A18A4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D83307B9-1E5C-DF11-B0C2-001D09F29146.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D80AD31E-0C5C-DF11-A14F-001D09F2983F.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D6A8A6B0-0F5C-DF11-B45A-000423D95030.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D6A301FA-155C-DF11-A9E8-000423D98BC4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D687FAFB-0E5C-DF11-962D-0030487A1FEC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D45E4694-145C-DF11-845A-000423D6B48C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D42F19EE-0C5C-DF11-9B38-003048D2C0F0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D4292519-FA5B-DF11-8D2C-001D09F2841C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D417FA8D-0D5C-DF11-AA3F-0016177CA7A0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D28C7501-015C-DF11-876C-001D09F251B8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D23B2FA8-1E5C-DF11-ACC1-001D09F251B8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D20DB330-1D5C-DF11-8D56-001D09F24934.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D0A96968-175C-DF11-B4CE-001D09F252DA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/D09A491B-0C5C-DF11-80EE-001617C3B6CE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CE996039-075C-DF11-87DC-003048D2C0F0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CE48B830-1D5C-DF11-804E-000423D98800.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CE14703B-075C-DF11-A72F-001D09F24691.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CCD2B922-245C-DF11-A42B-0030487C8CB6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CCB29621-FA5B-DF11-806E-0030487CD16E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CC8BD695-0D5C-DF11-AACA-001617DBCF6A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CC2A8A43-215C-DF11-A3E1-001D09F23A84.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CC19D8EF-F65B-DF11-8A1C-0030487CD718.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CADB5A1C-0A5C-DF11-B56C-000423D992DC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CA9CD7F3-F95B-DF11-81FF-0030487A322E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CA7A7470-255C-DF11-B89E-000423D998BA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/CA547598-FA5B-DF11-90A5-001D09F2A690.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C8EE1C3F-F65B-DF11-9479-000423D98750.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C88F2894-0D5C-DF11-AC7A-000423D9517C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C82C507D-195C-DF11-8B14-001617C3B706.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C6D382AC-085C-DF11-BAA2-000423D98EC4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C6ACC5A4-0D5C-DF11-BF6E-000423D944F0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C6836EED-215C-DF11-84D6-0030487CD7EA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C6512314-055C-DF11-89BB-001D09F28F0C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C607EF94-205C-DF11-A937-000423D99EEE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C60212BA-0F5C-DF11-8375-001617C3B69C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C4D9B129-265C-DF11-85F8-001D09F252E9.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C48FF08B-015C-DF11-8625-000423D9A212.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C44F5684-105C-DF11-98B2-001D09F2437B.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C4474A8E-015C-DF11-9CBF-000423D6006E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C43FB131-1D5C-DF11-AD51-0030487CD906.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C412ED8E-015C-DF11-8007-000423D99614.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C2D9869E-FC5B-DF11-9D11-000423D94524.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C263F217-245C-DF11-A8E1-0030487C6088.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C25F078A-195C-DF11-8212-0030487A3232.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C237F4BC-125C-DF11-8E09-001617C3B79A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C23311BC-1D5C-DF11-8E73-000423D95030.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C0FECEDF-0E5C-DF11-8413-0030487CD716.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C0DC066B-FF5B-DF11-8646-000423D6AF24.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C0B26E70-255C-DF11-962F-000423D98844.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C0754022-0C5C-DF11-9E57-0030487CD7C0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C05D7F5F-2A5C-DF11-B591-000423D98750.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/C033F60E-FC5B-DF11-8E03-0030487CD7CA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BEF7534F-FB5B-DF11-A22A-001D09F24763.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BE274798-205C-DF11-B6DC-000423D98B08.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BE163F14-FA5B-DF11-BB42-001D09F34488.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BE0FCBBC-125C-DF11-A228-001617C3B65A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BCB130A5-155C-DF11-AD13-003048D2C108.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BCAF2FA9-0F5C-DF11-96DB-001D09F24259.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BCA37338-075C-DF11-AACD-000423D952C0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BC9CC796-015C-DF11-87C7-001617E30D52.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BC387602-0A5C-DF11-9015-0019B9F704D6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BAF9D38C-0D5C-DF11-AE83-000423D99264.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BAA377C5-1D5C-DF11-AE4A-0016177CA7A0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BA9A6600-1B5C-DF11-B0ED-001617E30E28.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BA6C446C-F05B-DF11-A868-001D09F23A20.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/BA408FF8-F95B-DF11-A94E-003048678098.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B8AB89B7-1D5C-DF11-9B8D-000423D98F98.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B8504A36-115C-DF11-BC99-000423D992A4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B80086F6-285C-DF11-9A5A-0030487A18F2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B6F011D0-115C-DF11-A2AD-00304879FA4C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B6D43DD8-265C-DF11-A266-003048D2BF1C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B6B12297-1E5C-DF11-A6F2-001D09F2423B.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B69D754E-0F5C-DF11-9C7F-001D09F251B8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B649B4C6-065C-DF11-8E44-000423D6CA6E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B63C5E80-195C-DF11-BA76-0030487CF41E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B4F625B0-005C-DF11-81C0-001D09F253C0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B4D6221D-0C5C-DF11-A396-000423D991F0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B493F958-0A5C-DF11-BBD2-0030486780B8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B2F2809D-225C-DF11-810B-000423D94A20.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B2839BD8-1D5C-DF11-ADDD-001617C3B6E2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B0F96031-FA5B-DF11-877B-0019B9F72BFF.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B0BFBDE7-075C-DF11-BAEC-001D09F2B30B.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B0520BEE-0C5C-DF11-80DF-000423D9853C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/B0358A32-115C-DF11-A1FD-0030487C6088.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AEC4C5DA-0E5C-DF11-B3F6-0030487C90C4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AE89EBEE-F65B-DF11-AF55-0030487CAF0E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AE7467CD-065C-DF11-A564-000423D6B358.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AE32D361-2A5C-DF11-9744-0030487CD700.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/ACBF136D-175C-DF11-A337-0019B9F707D8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AC7DB240-F65B-DF11-8C8F-003048D2BDD8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AC325123-245C-DF11-BF09-001617E30F50.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AAFAC375-FD5B-DF11-853A-0030487A18D8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AAE43CB5-005C-DF11-B085-001D09F2AD4D.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AAD81A55-F85B-DF11-9A9F-0030487CD700.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AA597852-1C5C-DF11-986A-0030487C8E00.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AA42C937-115C-DF11-9EA0-0030487A3DE0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AA2DC649-215C-DF11-8732-001D09F2525D.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AA2A1917-FA5B-DF11-B2BF-001D09F253D4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/AA07ECBF-005C-DF11-AF70-0030487A322E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A8EB14E8-095C-DF11-BDC9-000423D6C8E6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A87B9B3D-075C-DF11-A8C9-001617C3B6C6.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A854907E-F35B-DF11-9440-0030487A1884.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A84CA4E2-2D5C-DF11-A30C-00304879BAB2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A8325338-075C-DF11-839F-001617C3B6CE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A8253832-265C-DF11-96E6-000423D99A8E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A81A76B9-165C-DF11-85B9-000423D987FC.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A6C37954-1C5C-DF11-8C75-0030487C90C2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A6783B29-265C-DF11-BCAC-000423D98E30.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A4D22D1C-0C5C-DF11-8990-000423D98750.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A48809A4-295C-DF11-BAE6-0030487CAF0E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A4377388-275C-DF11-B872-0030487C8E00.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A28D7FFD-155C-DF11-82BD-000423D99AA2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A2808601-FF5B-DF11-B696-001D09F2438A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A257F1EC-FE5B-DF11-B679-001D09F231B0.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A0EB45E2-095C-DF11-9A45-000423D98B5C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A0D024EE-0C5C-DF11-87CA-003048678098.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/A0C64D3B-FB5B-DF11-B751-001D09F2AD7F.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9E6ABC92-015C-DF11-8D93-000423D94494.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9E5AFBCF-FE5B-DF11-B657-0030487CD184.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9E54F05F-2A5C-DF11-A5AC-000423D98A44.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9E4D638C-195C-DF11-9CA3-0030487CD16E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9CE080EF-0C5C-DF11-A775-000423D6CA72.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9C12C190-125C-DF11-BF1E-000423D99614.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9A0C5349-075C-DF11-9035-000423D98800.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98F06B97-275C-DF11-8BD9-0030487A18F2.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98EFA9B0-125C-DF11-A1AB-001617C3B66C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98D67E12-FA5B-DF11-9966-001D09F2514F.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98CCA171-2C5C-DF11-B18E-000423D98E6C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98B05FAB-1E5C-DF11-A896-001D09F2983F.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98ABF70F-FA5B-DF11-93EF-0030487CF41E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/986458A2-1B5C-DF11-B708-001D09F252DA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98523408-0A5C-DF11-B991-000423D985E4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98185A72-FF5B-DF11-8E82-000423D174FE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9814C846-025C-DF11-93CF-0030487CD840.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/98066590-015C-DF11-8E90-000423D99264.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/965DB184-2A5C-DF11-8E39-0030487A18D8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/94EDA7CB-1D5C-DF11-8A80-000423D98DB4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/94603A12-FA5B-DF11-BEAA-001617DBD224.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/907922A7-445C-DF11-ACBE-001D09F29114.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9074E5B0-225C-DF11-81E2-0030487CD718.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/9004DCC1-125C-DF11-962C-003048D37560.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8EFA6455-FD5B-DF11-811E-0030487C6062.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8E9D5709-FA5B-DF11-ABD7-003048D2BE12.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8E86525A-1A5C-DF11-BD95-001617E30D40.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8E7516CB-005C-DF11-BE51-003048D2BC30.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8CF8089A-225C-DF11-A130-000423D6006E.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8CD97B6A-175C-DF11-9451-001D09F28F11.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8AF7C887-275C-DF11-A613-000423D998BA.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8A583DDA-0E5C-DF11-85CD-0030487C7828.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88EAD7FE-005C-DF11-9C65-001D09F297EF.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88E96750-FD5B-DF11-B5FB-0030487CD7EE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88CE50AE-085C-DF11-922F-001D09F24682.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88C516E8-075C-DF11-9620-001D09F2441B.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88A1A431-1D5C-DF11-BBCE-003048D2C108.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88951014-035C-DF11-9859-000423D94A04.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/888DAF8B-0D5C-DF11-9093-001617E30CE8.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/88702FAA-225C-DF11-A528-0030487CD7EE.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/8820B55A-235C-DF11-960A-000423D8FA38.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/880B19F0-0C5C-DF11-9C7B-000423D6B48C.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/86E49FAF-085C-DF11-9A85-0030487C90C4.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/86DAE2A2-225C-DF11-A371-0030487CD76A.root',
	'/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/175/867E8EDA-0E5C-DF11-A909-0030487C6088.root',
)
)

process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange(
   '135175:55-135175:561',
   '135175:563-135175:790',
   '135175:792-135175:1082',
)

# Conditions (Global Tag is used here):
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR10_P_V5::All'

#Geometry and field
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("TrackingTools.RecoGeometry.RecoGeometries_cff")

#tracker refitting -> trajectory
process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.ttrhbwr.ComputeCoarseLocalPositionFromDisk = True
process.generalTracks = process.TrackRefitter.clone(
   src = cms.InputTag("generalTracks")
)
process.ctfPixelLess = process.TrackRefitter.clone(
   src = cms.InputTag("ctfPixelLess")
)
process.refit = cms.Sequence(process.generalTracks*process.ctfPixelLess*process.doAlldEdXEstimators)
## re_fitting
#process.load('Configuration/StandardSequences/Reconstruction_cff')
#process.refit = cms.Sequence(
#    process.siPixelRecHits * 
#    process.siStripMatchedRecHits *
#    process.ckftracks *
#    process.ctfTracksPixelLess
#)

#analysis
process.analysis = cms.EDAnalyzer('TrackerDpgAnalysis',
   ClustersLabel = cms.InputTag("siStripClusters"),
   PixelClustersLabel = cms.InputTag("siPixelClusters"),
   TracksLabel   = cms.VInputTag( cms.InputTag("generalTracks"), cms.InputTag("ctfPixelLess") ),
   vertexLabel   = cms.InputTag('offlinePrimaryVertices'),
   pixelVertexLabel = cms.InputTag('pixelVertices'),
   beamSpotLabel = cms.InputTag('offlineBeamSpot'),
   DeDx1Label    = cms.InputTag('dedxHarmonic2'),
   DeDx2Label    = cms.InputTag('dedxTruncated40'),
   DeDx3Label    = cms.InputTag('dedxMedian'),
   L1Label       = cms.InputTag('gtDigis'),
   HLTLabel      = cms.InputTag("TriggerResults::HLT"),
   InitalCounter = cms.uint32(1),
   keepOntrackClusters  = cms.untracked.bool(True),
   keepOfftrackClusters = cms.untracked.bool(True),
   keepPixelClusters    = cms.untracked.bool(True),
   keepPixelVertices    = cms.untracked.bool(True),
   keepMissingHits      = cms.untracked.bool(True),
   keepTracks           = cms.untracked.bool(True),
   keepVertices         = cms.untracked.bool(True),
   keepEvents           = cms.untracked.bool(True),
#   DelayFileNames = cms.untracked.vstring(
#      "RND_TECMTHICK.xml",
#      "RND_TECMTHIN.xml",
#      "RND_TECPTHICK.xml",
#      "RND_TECPTHIN.xml",
#      "RND_TIB.xml",
#      "RND_TID.xml",
#      "RND_TOB.xml"),
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('trackerDPG.root')
)

process.skimming = cms.EDFilter("PhysDecl",
  applyfilter = cms.untracked.bool(False),
  debugOn = cms.untracked.bool(False),
  HLTriggerResults = cms.InputTag("TriggerResults","","HLT")

)

process.p = cms.Path(process.skimming*process.refit*process.analysis)
#process.dump = cms.EDAnalyzer("EventContentAnalyzer")
#process.p = cms.Path(process.refit*process.dump)

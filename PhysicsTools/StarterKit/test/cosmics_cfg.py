import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('PATLayer0Summary')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    PATLayer0Summary = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
     '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/02AC0387-1181-DD11-9E5E-0018F3D096EC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/0465ED82-1181-DD11-A198-001731A28585.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/04E10B9A-4A82-DD11-A57E-0018F3D096B4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/08F1A76D-7281-DD11-A542-001A92971BC8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/0E739F9A-1181-DD11-A433-001A928116C6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/12C68EAC-5E81-DD11-A84A-001A928116BA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/1450A3D5-7181-DD11-A145-00304867905A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/2044D99A-5D81-DD11-9533-001731AF68AD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/2CA21368-8781-DD11-BC7C-0018F3D09612.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/3A44CB8A-1181-DD11-BDB1-001A92811722.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/46EF728B-1181-DD11-B140-001A92971BCA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/46F10E10-1281-DD11-B00E-001A92810A98.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/5889EDE1-1181-DD11-9628-001A92811702.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/608DF2B4-8781-DD11-89C5-001A92971B1A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/60993087-1181-DD11-8D8F-001A928116EA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/6E917AA4-5D81-DD11-8BA4-003048767EDF.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/86E2AF9D-1181-DD11-8E79-001A92810AA2.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/9E1D024D-1281-DD11-A6A9-0018F3D09614.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/A04AEA75-8B81-DD11-8B97-003048767FB1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/A4B07224-7281-DD11-AB3B-003048678B0C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/AABFF396-3682-DD11-A17F-001A92810AF4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/ACF66D87-1181-DD11-ACBA-0018F3C3E3A6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/B4811605-1281-DD11-B53D-001A92811718.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/C0852D05-7281-DD11-9639-001A928116FC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/D8E5E811-4A82-DD11-A2AE-001BFCDBD1BC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/DABEC93A-7081-DD11-9255-003048678BAE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/DECE279F-7181-DD11-92AD-00304875A9FD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/E0BFE4EE-7181-DD11-9437-003048D15DF6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0000/F07FAD7F-1181-DD11-91B6-0018F3D096C8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/02100BCE-2583-DD11-B3A9-001A92971B9A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/0A6F09C6-2C83-DD11-92F0-001A92971BB8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/0C87C2BB-D683-DD11-BB27-003048678A76.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/106C8AD9-2B83-DD11-B490-0030486791BA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/125A9591-4683-DD11-B61C-00304875ABEF.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/126022DC-2C83-DD11-8573-003048679000.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/1616385D-D183-DD11-A957-00304867900C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/1C7286A4-2F83-DD11-8005-001731AF65E9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/1CDDA0C9-2583-DD11-BB49-0018F3D09692.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/22846A58-2A83-DD11-B989-003048679266.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/22D926D6-D183-DD11-8426-00304875F317.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/24A9E89E-2583-DD11-B1D5-001731A283E1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/283C4CE2-4683-DD11-8DD6-003048678A88.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/2A44168E-F883-DD11-AF18-001731A28319.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/2A5225B9-0584-DD11-8934-001BFCDBD184.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/2A830D32-D383-DD11-83FA-003048767DC7.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/30F4B584-2F83-DD11-A686-001A9281173E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/30F76DB9-D583-DD11-A547-003048754C51.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/3472C895-2F83-DD11-9CA4-003048678BAC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/34EA0062-2B83-DD11-ACBD-00304867926C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/3E83CF17-0384-DD11-8BD6-001A92811708.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/3EB4F0A0-2D83-DD11-9918-001BFCDBD11E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/408F844C-D983-DD11-A33C-003048679000.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/42AE7999-2B83-DD11-8B54-003048678BE8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/4641ED1D-FB83-DD11-BC3D-0017312A247F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/466538E6-0484-DD11-BFC7-0018F3D09706.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/46781917-0384-DD11-A050-001A92810AD4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/48DBD30F-FF83-DD11-97B9-001A92811724.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/4A3196C1-C483-DD11-900F-001731AF698F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/4C005929-FB83-DD11-A1D9-00304875ABE1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/4EDF74CD-2383-DD11-806E-0018F3D096B6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/5208A389-2F83-DD11-A4F6-0018F3D0969C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/521BA1E3-4683-DD11-8B7D-003048679000.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/5AB533A4-0584-DD11-B1B8-00304875AAE9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/5C196D97-2383-DD11-AF19-0018F3D09612.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/607860CF-AD88-DD11-B47F-00304875A9E9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/6265D775-AB88-DD11-822D-001A92810ACA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/70337529-4783-DD11-9965-003048678B18.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/72743378-2E83-DD11-91A3-001731AF66EF.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/74229FFF-FD83-DD11-B16C-003048767E4B.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/76360BE5-4683-DD11-B9D5-003048D15E2C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/76369838-2A83-DD11-9C9F-003048678CA2.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/76B1F2F5-4683-DD11-9E38-003048678ED2.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/76E2C048-AE88-DD11-B8AD-00304875ABE1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/783763A3-2D83-DD11-BB97-003048678ADA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/7CEB34E0-F783-DD11-A269-001731AF66BB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/7EC02923-D088-DD11-BE83-001A92810AA2.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/841B04B0-0284-DD11-8154-001A92971B8C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/8AA2F620-C283-DD11-BDAF-003048678B16.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/8E96EDBE-BC83-DD11-9DDD-003048D15E24.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/94134F69-2E83-DD11-A043-003048678BAC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/946AF476-D383-DD11-8A7D-001A92971B3C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/9AC7B82B-2583-DD11-B2AD-001731AF66A5.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/9AD0D23E-FD83-DD11-91D8-001731AF66BB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/9CDE2EB3-D183-DD11-9736-001731AF6B77.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/9E386CBF-D683-DD11-BDCB-00304876A061.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/A6A95CDB-2983-DD11-A8C8-003048678BB8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/A8361D36-2A83-DD11-82D3-003048679244.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/AC45819D-2D83-DD11-B3F5-003048678BB8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/B03DF9CC-9284-DD11-A320-003048756485.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/B212A5B0-0484-DD11-8861-001BFCDBD11E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/B2467FB9-2583-DD11-AB64-0018F3D09704.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/B2C6F7DA-C483-DD11-883B-001BFCDBD100.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/B4A4CF4B-C483-DD11-81E2-001731AF698F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/B63CA3B4-2F83-DD11-A939-001731AF65E9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/BA8B0A65-D383-DD11-BD00-00304875A587.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/BAC5DE76-D683-DD11-BA9A-003048679274.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/BADE3D6B-2E83-DD11-8A9A-003048678D52.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/BCE831A8-4683-DD11-91B4-001A92971B8E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/BE0F0439-4783-DD11-B4AD-003048678F84.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/C2A10FC1-AE88-DD11-8EB8-001A92971BDA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/C64E7813-2A83-DD11-8962-003048D15E24.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/C66EE0B9-2383-DD11-942A-001A92971B08.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/C6AFEC38-2A83-DD11-A0AF-003048678C3A.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/CEB86A0D-9384-DD11-9E68-00304875ABED.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/D209BF89-AA88-DD11-ACBD-001A92810AE6.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/D261231F-0884-DD11-9C08-00304875AC63.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/D270A9A2-D183-DD11-A0E8-00304875A339.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/D47D691D-4783-DD11-AB34-003048D15CC0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/D89C34CB-2383-DD11-A84E-0018F3D095EA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/DA0220C1-2583-DD11-921E-001731AF66A5.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/DE9B3573-2A83-DD11-A65D-001731AF65E9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/E0663F72-1D83-DD11-92B7-0018F3D0970C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/E2D19150-2583-DD11-8A2D-0018F3D09692.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/E2F1893D-2A83-DD11-841B-0030486792F0.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/E43243DF-0384-DD11-8F27-001A92971ADC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/E8FA7FA6-AC88-DD11-A7C5-003048756485.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/EAA6BFD7-2483-DD11-B9F5-0018F3D0970C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/EC012A95-D683-DD11-B832-003048767DF9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/EC396662-D183-DD11-8D3D-00304875A339.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/ECBC8617-AB88-DD11-97FD-003048767E59.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/EE8BAC17-0484-DD11-9ABB-0018F3D09700.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0001/F8A2E611-FF83-DD11-A729-00304875A7B1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/1A960256-0F86-DD11-B2EA-00304875ABF1.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/20581653-0E86-DD11-993D-003048754FBB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/2220D972-F786-DD11-B3E7-001731AF65E9.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/284FDB52-0E86-DD11-A128-003048754C5F.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/2A00795C-0E86-DD11-A743-003048767E5B.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/2E421F24-0F86-DD11-9630-00304875AABD.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/307DDBB7-1886-DD11-A5AB-003048754FBB.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/32E3EBF3-AD87-DD11-BE0A-0018F3D096DA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/40C07E44-6485-DD11-B364-001BFCDBD130.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/4C78879F-6485-DD11-8D2E-0018F3D096BE.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/5088264C-B787-DD11-95F1-0018F3D0970C.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/681CAADB-B187-DD11-A9E8-0018F3D09642.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/747397AD-0E86-DD11-B9F3-00304875A7AF.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/76D8ED6F-F786-DD11-9B51-001A92971B38.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/7A5CBF40-AE87-DD11-BD00-001731A28543.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/7AFE71C4-6485-DD11-A35F-0018F3D09628.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/94168958-6985-DD11-A632-0018F3D096DC.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/982B78C3-6485-DD11-9AF8-001A92810AE4.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/A0691869-6585-DD11-AD74-0018F3D096DA.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/AA9698A8-6485-DD11-AD4D-0018F3D09664.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/C4EDA7B5-1786-DD11-B06B-001731AF67EF.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/C68CC29F-6885-DD11-9E7D-00304867902E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/D4701324-0F86-DD11-8FE3-00304875AA15.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/D6107C9C-0E86-DD11-82AA-00304876A137.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/D67C59AD-0E86-DD11-BE0F-00304875A9E5.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/D6EE3476-F786-DD11-B336-001731AF6867.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_multiCosmicMuon_v1/0002/EE53FFB4-1786-DD11-ACE9-001731A28BE1.root'
     )
                            )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# Load PAT Layer 0+1 configuration
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")

### Custom PAT sequences
# Reconstruction sequence: no electron, no b-tagging (needs vertex), no photon ID
process.patBeforeLevel0Reco_withoutPFTau.remove(process.patAODBTagging)
process.patBeforeLevel0Reco_withoutPFTau.remove(process.patAODElectronIsolation)
process.patBeforeLevel0Reco_withoutPFTau.remove(process.patAODPhotonID)


# Change endcap cluster names
process.egammaSuperClusterMerger.src = cms.VInputTag(
    cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters'),
    cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
    )
process.egammaBasicClusterMerger.src = cms.VInputTag(
    cms.InputTag('cosmicBasicClusters','CosmicBarrelBasicClusters'),
    cms.InputTag('cosmicBasicClusters','CosmicEndcapBasicClusters')
    )
    

# Cleaning sequence: no electrons
process.patLayer0Cleaners_withoutPFTau.remove(process.allLayer0Electrons)

# High level reco: no electrons, no generator information, or vertexing
process.patHighLevelReco_withoutPFTau.remove(process.patElectronId)
process.patHighLevelReco_withoutPFTau.remove(process.patLayer0ElectronIsolation)
process.patHighLevelReco_withoutPFTau.remove(process.patLayer0PhotonID)
process.patHighLevelReco_withoutPFTau.remove(process.patJetFlavourId)          # Needs MC
process.patHighLevelReco_withoutPFTau.remove(process.patLayer0BTagging)        # Needs vertex
process.patHighLevelReco_withoutPFTau.remove(process.patLayer0JetTracksCharge) # Needs vertex

# No truth matching
process.patLayer0_withoutPFTau_withoutTrigMatch.remove(process.patMCTruth_withoutTau)

# Cross-cleaning: no electrons
process.allLayer0Jets.removeOverlaps = cms.PSet()
process.allLayer0Photons.removeElectrons = cms.string('none')

# Replace muon tracks' input labels
process.muIsoDepositTk.ExtractorPSet.inputTrackCollection = "cosmicMuons"
process.gamIsoDepositTk.ExtractorPSet.inputTrackCollection = "cosmicMuons"

# Layer 1: no electrons, no hemispheres, no taus
process.patLayer1.remove(process.layer1Electrons)
process.patLayer1.remove(process.layer1Taus)
process.patLayer1.remove(process.countLayer1Leptons)
process.patLayer1.remove(process.layer1Hemispheres)


### Tune producers (remove non-existing associated information)
process.allLayer1Muons.addResolutions = False
process.allLayer1Muons.addTrigMatch   = False
process.allLayer1Muons.addGenMatch    = False

process.allLayer1Photons.addGenMatch  = False
process.allLayer1Photons.addTrigMatch = False
process.allLayer1Photons.addPhotonID  = False

process.allLayer1Jets.addAssociatedTracks = False
process.allLayer1Jets.addJetCharge        = False
process.allLayer1Jets.addTagInfoRefs      = False
process.allLayer1Jets.addDiscriminators   = False
process.allLayer1Jets.addTrigMatch        = False
process.allLayer1Jets.addGenPartonMatch   = False
process.allLayer1Jets.addGenJetMatch      = False
process.allLayer1Jets.getJetMCFlavour     = False

process.allLayer1METs.addResolutions      = False
process.allLayer1METs.addTrigMatch        = False
process.allLayer1METs.addGenMET           = False


process.p = cms.Path(
      process.patMuonIsolation # Add missing muon isolation information
    * process.egammaSuperClusterMerger * process.egammaBasicClusterMerger
    * process.patAODPhotonIsolation
    * process.patLayer0_withoutPFTau_withoutTrigMatch
    * process.patLayer1
    )


# Output module configuration
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('patcosmics.root'),
    # save only events passing the full path
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *')
)
process.outpath = cms.EndPath(process.out)
# save PAT Layer 0 output
process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
process.out.outputCommands.extend(process.patLayer1EventContent.outputCommands)


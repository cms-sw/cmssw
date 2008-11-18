import FWCore.ParameterSet.Config as cms

process = cms.Process("ClusterThr")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('ClusterThr'), 
                                         
    debug = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('debug')
    )

#-------------------------------------------------
# Magnetic Field
#-------------------------------------------------
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#-------------------------------------------------
# CMS Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# CMS Service
#-------------------------------------------------
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('testClusThr.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )
                                 
#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load("AnalysisExamples.SiStripDetectorPerformance.ClusterThr_cfi")

#-------------------------------------------------
#TrackRefitter With Material
#-------------------------------------------------
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
# from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
process.TrackRefitter.src  = 'ctfWithMaterialTracksP5'
process.TrackRefitter.TrajectoryInEvent = True


#-------------------------------------------------
# In-/Output
#-------------------------------------------------

# input
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
  '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/025C2F1E-33A1-DD11-ADD9-0030487A1990.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/0439D4C8-2CA1-DD11-9D73-001617C3B6FE.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/060149C5-28A1-DD11-9545-001617E30D52.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/06AE0637-24A1-DD11-B86B-001D09F2910A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/08A444EA-30A1-DD11-AE8B-001D09F24EAC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/08A81A29-40A1-DD11-835C-001D09F231B0.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/08E7F061-33A1-DD11-A731-001617C3B66C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/0AE0F3B9-2DA1-DD11-98A3-000423D951D4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/0C490871-3DA1-DD11-AE02-000423D94534.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/0C8736F6-2DA1-DD11-A6E7-001D09F24682.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/0C8973FA-3AA1-DD11-BBAE-001D09F24FBA.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/0E6BE537-39A1-DD11-A738-000423D991D4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/102917C4-1AA1-DD11-A5AB-001D09F23174.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/105DAE90-32A1-DD11-861C-001617E30D0A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/1218F672-28A1-DD11-AE13-001D09F241B9.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/12B173FD-3BA1-DD11-9BA2-000423D99264.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/141D0FFC-24A1-DD11-B761-000423D952C0.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/162FC2A2-3EA1-DD11-A5AA-001617E30D38.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/18616231-27A1-DD11-B632-001D09F29538.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/1A773ECB-34A1-DD11-8F6A-001617E30F48.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/1ABD0F68-17A1-DD11-8CD7-001D09F29524.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/1ABDA398-25A1-DD11-B7E2-0019B9F72BAA.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/1EA21113-31A1-DD11-AAA1-0030487A3232.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/1ED56C6C-17A1-DD11-B6BA-0030487D1BCC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/202F348B-1BA1-DD11-81CF-000423D944DC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2057A287-2EA1-DD11-A5EB-000423D94534.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2253FA45-2EA1-DD11-8722-000423D98B5C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/227B6ED7-41A1-DD11-B832-000423D98844.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/24E355D7-1DA1-DD11-AA29-000423D6CA02.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/28F2EDC9-25A1-DD11-BEEE-001D09F24682.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2A1496BB-30A1-DD11-9CBE-0019DB2F3F9B.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2A968010-47A1-DD11-922A-000423D9939C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2A97F320-2CA1-DD11-9280-000423D6CA42.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2ABF30CC-25A1-DD11-8779-001D09F29849.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2AE58205-24A1-DD11-98C7-001D09F23174.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/2CEDF310-3EA1-DD11-8ADA-0030487A3232.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/30750E43-24A1-DD11-B3AD-0030487A3C9A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/30DFCE5C-30A1-DD11-8084-000423D6A6F4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/3256478B-3DA1-DD11-8782-000423D9997E.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/344C6C77-1BA1-DD11-A391-001617C3B778.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/3476566C-3FA1-DD11-897E-000423D174FE.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/3685A096-3CA1-DD11-912E-001617E30D00.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/36B05EBB-33A1-DD11-9563-001D09F290BF.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/38121BEE-35A1-DD11-B0B5-001D09F28F11.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/38DA30FA-2DA1-DD11-BCAE-00304879FA4A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/3A1F709B-1FA1-DD11-B56C-001617E30CC8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/3C46EBB0-38A1-DD11-8BDC-001D09F24353.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/40AC738D-19A1-DD11-90E4-001D09F2A49C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/42121380-29A1-DD11-859C-001D09F23F2A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/423A8D91-2AA1-DD11-A64B-000423D94700.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4286A578-25A1-DD11-B373-000423D985B0.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4410CB89-1CA1-DD11-8F6B-001D09F26C5C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/44E73D73-2FA1-DD11-968D-000423D98634.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4620F27F-3DA1-DD11-9F38-000423D94524.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/487E13B2-39A1-DD11-81BD-001D09F24D67.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/48C96248-40A1-DD11-95BC-001617E30CA4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4A2D8D78-2AA1-DD11-A5BF-001617E30CC8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4A8CA58C-1CA1-DD11-83A8-0019B9F72CE5.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4AB2E359-27A1-DD11-8E7A-001D09F290BF.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4C04CD30-31A1-DD11-B6AF-000423D6CA72.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4C091369-1CA1-DD11-A7E3-001D09F23A61.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4CAD1FA1-1AA1-DD11-9810-0030487A18A4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4CB5CD76-39A1-DD11-97C8-001D09F29533.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4CCF6CE5-2AA1-DD11-9437-001D09F24691.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/4EC94D75-2CA1-DD11-9800-001D09F25217.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/50370527-21A1-DD11-8BA5-001617E30CD4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/56ED343C-1BA1-DD11-BD2F-001617E30D52.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/58550EB2-2AA1-DD11-89DA-001617C3B69C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/58D5F7A0-3BA1-DD11-B45E-000423D99AA2.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/58D63078-2AA1-DD11-8D49-001617DBD316.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/5A0FCF52-26A1-DD11-8493-001617E30F50.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/5CBBB8FD-24A1-DD11-9831-000423D9A212.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/5CE17A1B-3DA1-DD11-A093-001617C3B76A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/5CFB8669-23A1-DD11-98E5-0030487A1FEC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/5EAF44FC-35A1-DD11-99A8-001617DBCF1E.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/605AD70E-3EA1-DD11-9A1D-001D09F253D4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/60F56090-1FA1-DD11-8B44-000423D174FE.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/6295CFF9-2CA1-DD11-9629-001D09F2503C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/6623D49B-1CA1-DD11-B0FC-000423D98BE8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/666FE4BD-16A1-DD11-9179-001D09F2841C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/68698A42-1DA1-DD11-B590-000423D6C8EE.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/688C0D32-19A1-DD11-99C9-001D09F24FEC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/68FC02A1-30A1-DD11-876C-001D09F2424A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/6C71F0E0-2AA1-DD11-A4AC-001D09F253C0.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/706337D5-26A1-DD11-B989-000423D98950.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/726CC08E-1AA1-DD11-8AA6-001D09F24664.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/72C63FEA-30A1-DD11-916A-001D09F28C1E.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/72F87DD3-31A1-DD11-AC4E-001617C3B6E8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/74422B5D-36A1-DD11-A8DB-001D09F25479.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/768AE794-2FA1-DD11-B4BB-001D09F24047.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/789527FE-19A1-DD11-88BC-000423D99F1E.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/78A7CA7D-3FA1-DD11-BE96-000423D94E70.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/7A27B19E-24A1-DD11-B1D9-001617E30F4C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/7A9FEBB8-3EA1-DD11-BBC7-001617C3B5E4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/7AD2F27F-29A1-DD11-AA4C-001D09F244DE.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/7E8500BF-3EA1-DD11-A891-000423D99E46.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/7EF1D09A-26A1-DD11-A1FE-001D09F23A20.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/80784E5C-31A1-DD11-937D-001617E30F50.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/80B1F575-28A1-DD11-A74C-001D09F25479.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/80BD599D-3BA1-DD11-A36F-000423D98B6C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8254CAB2-3FA1-DD11-9954-0019DB2F3F9B.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/82D095E7-1DA1-DD11-A266-000423D992DC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/845EE97E-2FA1-DD11-B079-001D09F2AD84.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/84C91FF8-41A1-DD11-BD94-000423D99394.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/84E904CB-32A1-DD11-A623-000423D98920.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/86A537F7-2EA1-DD11-8E81-001617DF785A.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8AB47D9D-2FA1-DD11-87B1-001617C3B6DC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8C1FDAD3-2FA1-DD11-8F13-0019B9F7312C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8C90896B-3BA1-DD11-952D-000423D99AA2.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8CE46871-2AA1-DD11-AB99-000423D98EC8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8E8C99BB-37A1-DD11-9EB4-001D09F290BF.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/8EA12107-27A1-DD11-8B3E-000423D9890C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/90C9905E-36A1-DD11-AD6B-001D09F2503C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/928E8872-22A1-DD11-B00D-000423D99BF2.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/9496331D-36A1-DD11-A52C-000423D98E6C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/94AC8BEA-27A1-DD11-8F4A-001D09F25393.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/98785958-2BA1-DD11-8E1D-001D09F23A3E.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/9CF64820-42A1-DD11-82C4-000423DD2F34.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/9E5CD882-2EA1-DD11-951E-001617E30D12.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/9EB37A81-36A1-DD11-9C9F-001617C3B654.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/9EC36DB6-2BA1-DD11-B98E-001D09F26C5C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A025EF9A-34A1-DD11-88C9-000423D985E4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A0743792-37A1-DD11-BAD3-000423D98AF0.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A077CB6C-2FA1-DD11-8A18-001617C3B5E4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A0CA8D03-29A1-DD11-89D5-0030487A18A4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A0D0161C-3BA1-DD11-9836-001D09F231B0.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A2592F90-19A1-DD11-A4F9-001D09F2426D.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A2D39DE1-2AA1-DD11-AB4A-001D09F29849.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A2DA7479-1EA1-DD11-9D33-000423D6B2D8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A4AAA37E-2CA1-DD11-9298-001D09F242EA.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A627B3D1-27A1-DD11-B160-001D09F25438.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/A8DC8286-1EA1-DD11-B066-000423D6AF24.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/AA751F01-1CA1-DD11-9166-001617C3B5F4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/AA956E86-2CA1-DD11-BA29-000423D98834.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/ACDDA90D-3EA1-DD11-BA91-001D09F24493.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/AE71D68B-1BA1-DD11-9FAC-001617C3B6DE.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B231141A-36A1-DD11-8C09-0030487A18A4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B259AA3C-37A1-DD11-A1D7-000423D6B444.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B2C75DFE-35A1-DD11-8D6F-001D09F297EF.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B2DE193A-18A1-DD11-B668-000423D986C4.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B2EA0EEB-33A1-DD11-A845-000423D944DC.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B6A0CD78-37A1-DD11-87F8-001617C3B5D8.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/B86CB8B6-4FA1-DD11-A700-001617C3B66C.root',
        '/store/data/Commissioning08/Cosmics/RECO/v1/000/067/173/BA055C49-33A1-DD11-B902-000423D99394.root'   
    )
                            )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100000))

process.ctfWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                          src = cms.InputTag('ctfWithMaterialTracksP5'),
                                                          minNumber = cms.uint32(1) 
                                                          )
#-------------------------------------------------
# Scheduling
#-------------------------------------------------

process.outP = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.ctfWithMaterialTracksP5TkCntFilter*process.TrackRefitter*process.ClusterThr)
process.pout = cms.EndPath(process.outP)

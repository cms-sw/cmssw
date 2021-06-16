import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripHotStripCalibration")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring(
        'orbitFilter', 
        'OrbitFilter', 
        'SiStripQualityHotStripIdentifierRoot', 
        'siStripQualityHotStripIdentifierRoot', 
        'SiStripHotStripAlgorithmFromClusterOccupancy'
    ),
    files = cms.untracked.PSet(
        log = cms.untracked.PSet(
            extension = cms.untracked.string('txt')
        )
    ),
    log = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


# Exclude masking #

from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import *
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string( 'SiStripDetCablingRcd' ), tag = cms.string( '' ) )
#           cms.PSet( record = cms.string( 'SiStripBadChannelRcd' ), tag = cms.string( '' ) )

       )


#-----------------------
# Reconstruction Modules
#-----------------------

process.load("DQM.SiStripMonitorClient.RecoForDQM_Cosmic_cff")

#--------------------------
# DQM
#--------------------------
process.load("DQM.SiStripMonitorClient.SiStripDQMOnline_cff")

process.SiStripMonitorCluster.TH1ClusterPos.moduleswitchon = cms.bool(True)
process.SiStripMonitorCluster.TH1nClusters.layerswitchon  = cms.bool(True)
process.SiStripMonitorCluster.TH1ClusterStoN.moduleswitchon = cms.bool(True)
process.SiStripMonitorCluster.TH1ClusterStoNVsPos.moduleswitchon = cms.bool(True)

process.SiStripMonitorCluster.TH1ClusterNoise.moduleswitchon = cms.bool(False)
process.SiStripMonitorCluster.TH1NrOfClusterizedStrips.moduleswitchon = cms.bool(False)
process.SiStripMonitorCluster.TH1ModuleLocalOccupancy.moduleswitchon = cms.bool(False)
process.SiStripMonitorCluster.TH1nClusters.moduleswitchon = cms.bool(False)
process.SiStripMonitorCluster.TH1ClusterCharge.moduleswitchon = cms.bool(False)
process.SiStripMonitorCluster.TH1ClusterWidth.moduleswitchon = cms.bool(False)

#--------------------------


from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        #available filetypes: .pdf .png .jpg .svg
        TkMapFileName = cms.untracked.string("TkMapBadComponents.png")
        )


process.p = cms.Path(process.siStripDigis*process.siStripZeroSuppression*process.siStripClusters*process.SiStripMonitorCluster*process.dqmSaver*process.stat)

#-------------------------
# Input Events
#-------------------------

process.source = cms.Source("PoolSource",
                            timetype = cms.string('runnumber'),
                            firstRun = cms.untracked.uint32(67173),
                            lastRun = cms.untracked.uint32(67173),
                            #skipEvents = cms.untracked.uint32(4000000),
                            interval = cms.uint32(1),
                            fileNames = cms.untracked.vstring(
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/26A71E49-13A1-DD11-B75B-001617C3B706.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/00202978-0EA1-DD11-9497-001D09F25438.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/160B9F74-0EA1-DD11-969D-001D09F29597.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/D2CC5FE9-14A1-DD11-A2EE-000423D98B5C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/D06E060F-12A1-DD11-A95D-00304879FA4C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/0AE39292-12A1-DD11-88AC-0030487A3232.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/BEB52342-0FA1-DD11-A52A-001D09F26C5C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/CC5DE1FA-14A1-DD11-9F9C-001617C3B65A.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7C5241CC-16A1-DD11-B5C5-001D09F28F1B.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/80286751-19A1-DD11-B8DA-000423D98EA8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C0E9A70F-19A1-DD11-9165-001D09F29597.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/02DE35FE-15A1-DD11-B662-001D09F2A465.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/4A37177E-18A1-DD11-9012-0030487A1990.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/DAB9DC03-1AA1-DD11-BFED-001D09F25479.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/D2928A55-26A1-DD11-A9B1-000423D98E6C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3E0DA7BA-22A1-DD11-A5A2-001D09F24682.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/66BA4CF0-26A1-DD11-9831-00304879FA4A.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C4CA4D77-1CA1-DD11-A8F0-000423D6CA42.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/447F9655-2FA1-DD11-A70C-001D09F2AF1E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/88FEB544-35A1-DD11-9B4A-001617E30F4C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2ADB7094-33A1-DD11-BC76-001D09F28755.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/BE5CCC91-33A1-DD11-9229-0019B9F70468.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/AABE9D37-37A1-DD11-92B8-001D09F25442.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/4C3704DB-35A1-DD11-8C7D-0030487D1BCC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A27D0992-32A1-DD11-8FC1-001D09F28D4A.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3E78979B-36A1-DD11-B312-000423D952C0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A69A807A-2FA1-DD11-B4BF-001D09F25401.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3A593D20-34A1-DD11-A171-000423D8F63C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/964C38EF-24A1-DD11-9FAF-001D09F28C1E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C864B2C9-26A1-DD11-9230-001D09F24EAC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C48565C1-27A1-DD11-9229-0030487D1BCC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E63E936B-29A1-DD11-8A6D-001D09F2437B.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/DCD97C68-2AA1-DD11-B301-001617DBD5AC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/44851555-37A1-DD11-B1BD-001D09F290BF.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3EB3749A-10A1-DD11-9CD8-001D09F24EE3.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7630D61B-1EA1-DD11-ABA3-001617C3B654.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/400083E4-18A1-DD11-8460-001D09F24303.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/6EA2F153-1AA1-DD11-A86C-0019B9F72CE5.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/20E248D6-1CA1-DD11-AE55-001617DBD5B2.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/9076E132-1EA1-DD11-BD14-001617DBD288.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/D4650D84-18A1-DD11-9D37-001D09F24EC0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/04C9406B-1AA1-DD11-85A5-001D09F241F0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/AE8CA836-1FA1-DD11-BB6E-001D09F297EF.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/760EF1F3-24A1-DD11-BB92-0030487A3232.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7809AD75-30A1-DD11-917E-001D09F24498.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A6283824-2CA1-DD11-A0F6-001D09F24EAC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/5A75B407-2BA1-DD11-B928-001D09F241B4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/B2CE11E4-30A1-DD11-94E5-001D09F24DDF.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7CB8AA60-2BA1-DD11-905C-001D09F26C5C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/00B9C6C5-2FA1-DD11-A057-001617C3B70E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/F0F19E8F-32A1-DD11-9616-000423DD2F34.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7EF1786E-37A1-DD11-8EE4-001D09F24353.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/80C578EC-33A1-DD11-A922-0019B9F730D2.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A4486C59-31A1-DD11-A515-001D09F250AF.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/46711319-23A1-DD11-9848-001D09F28F11.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1E89DD50-27A1-DD11-9F12-001D09F23A07.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7AFA19FB-27A1-DD11-9A22-000423D98E54.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/96C67470-29A1-DD11-B822-001D09F254CE.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1092439A-22A1-DD11-B2AC-001D09F28F11.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/826C0559-23A1-DD11-A48A-001D09F2305C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2C16305A-4EA1-DD11-B6DC-001617C3B5F4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1AF6AB4A-0FA1-DD11-B1EF-001617C3B5E4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/861FDFF0-13A1-DD11-B9C0-001D09F23A4D.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/54CA2A35-1AA1-DD11-B8D6-0019B9F730D2.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/D6288E75-0EA1-DD11-9921-000423D98EA8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/EE87300D-17A1-DD11-B454-001D09F24498.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7802E037-15A1-DD11-AA43-0019B9F71A6B.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/B85B58DF-1AA1-DD11-8E78-0019B9F7312C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/40B3630C-21A1-DD11-8845-001D09F23C73.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/0C05E4BB-1EA1-DD11-B0F6-000423D98B08.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/84AFD0D6-2BA1-DD11-9A7E-0019B9F704D6.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/9EB7C042-2BA1-DD11-93ED-001D09F248FD.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/EA22A649-32A1-DD11-910B-001D09F2516D.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E47EF413-37A1-DD11-9EC5-001D09F23C73.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/76BA6EF1-2FA1-DD11-A0D0-001D09F2516D.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/201008D0-35A1-DD11-85B8-001D09F23A20.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/96EFB098-28A1-DD11-A719-0019B9F707D8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/66D89A86-24A1-DD11-A9CC-001D09F25479.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/061BB1B6-2DA1-DD11-8C5A-001617DBCF6A.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E0233EBF-28A1-DD11-B3E4-001D09F28755.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/9C2ED07D-25A1-DD11-8D45-001D09F24024.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/CC093B29-13A1-DD11-A4E1-001617C3B76E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/94341F3E-0EA1-DD11-A1DD-000423D9853C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1A07C3A1-17A1-DD11-A21A-001617E30CA4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A6295BC2-0FA1-DD11-86F7-001D09F29533.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/0CCA0F69-21A1-DD11-BDCD-0030487D1BCC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/6E3B43DB-1BA1-DD11-A351-001D09F2A49C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E4B97394-1FA1-DD11-9624-001D09F2905B.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/78998D0A-21A1-DD11-A3FD-001D09F282F5.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3475B992-20A1-DD11-A79F-0030487A3232.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7C9EF0DA-1BA1-DD11-B2A2-001D09F26C5C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/EED82124-2CA1-DD11-9BA8-001D09F25109.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C0922ED1-2CA1-DD11-89D9-001D09F242EF.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A62D5110-36A1-DD11-BAC9-001617C3B710.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/BCD2B360-2CA1-DD11-8681-001D09F23174.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/223A0911-36A1-DD11-98A4-0030487A322E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/BC0990DE-22A1-DD11-9ACC-001D09F23C73.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/128820BE-29A1-DD11-9F94-000423D94A20.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/8C830BE2-13A1-DD11-800A-0030487A1FEC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/68926618-10A1-DD11-9C38-000423D98EA8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/FA93D4B6-1FA1-DD11-BFC4-0030487A3232.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7058E29A-11A1-DD11-B46F-001D09F2AD4D.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7639A432-17A1-DD11-9A53-001D09F2960F.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A0CB32DA-15A1-DD11-B2D7-0019B9F72D71.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/4CA0CFAC-16A1-DD11-8D31-001D09F29538.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/42859B66-1FA1-DD11-AA76-0030487C6090.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2CB2367D-1FA1-DD11-9803-001D09F2AF1E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A0E00D8E-21A1-DD11-B4C9-001D09F28F11.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/021F2341-22A1-DD11-AC42-001617DF785A.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/8A577B25-1BA1-DD11-B42D-001D09F2A465.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/84D9A3C7-33A1-DD11-8657-00304879FA4C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/B4ABA274-30A1-DD11-ACDB-0019B9F70607.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1A53C081-2AA1-DD11-8F77-000423D6B444.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E4E71E26-32A1-DD11-B723-001D09F24047.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C653F78E-34A1-DD11-B786-001D09F28F11.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E6416335-37A1-DD11-9CCF-0030487C6090.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/04406232-33A1-DD11-894D-001D09F252E9.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/DA5B5F9C-25A1-DD11-B7E5-001D09F231C9.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3A004727-27A1-DD11-A6DA-0019B9F70468.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1AAF8407-20A1-DD11-9887-001D09F23A20.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/AE1CF09C-27A1-DD11-9B69-001D09F290D8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/AE0BE777-2AA1-DD11-A2E2-000423D95220.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/F4730EF1-12A1-DD11-816A-000423D951D4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2812FE4B-0FA1-DD11-8ABB-001D09F2841C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1C517CF1-0DA1-DD11-81A8-001D09F29533.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/B0BD148D-10A1-DD11-B38B-0019B9F72D71.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/226A8FC0-11A1-DD11-97E0-000423D944F8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/0A96DAF0-13A1-DD11-9E9A-001D09F24FE7.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2C23E398-1DA1-DD11-8FEE-001617DBD472.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1A2889A5-18A1-DD11-BAC0-001D09F25208.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1CFCE413-1EA1-DD11-BDC0-001617E30F50.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/F6E5D27A-19A1-DD11-BF68-001D09F242EA.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/8E6F41ED-25A1-DD11-922F-001617C3B6DE.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/7E304D31-20A1-DD11-8E80-0019B9F70468.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2E7E9903-22A1-DD11-BA3F-000423D98634.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/B6F4E609-2FA1-DD11-B4F9-001D09F231B0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/406F06EE-30A1-DD11-8772-001617DBD230.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/E0D42E55-2DA1-DD11-9D37-000423D98800.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/18A59666-33A1-DD11-9167-000423D6B358.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3263F89C-31A1-DD11-AD1E-001D09F24763.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/5EC922EB-30A1-DD11-8896-001D09F24691.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/DE44F47C-22A1-DD11-B2CA-000423D99B3E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/6A968981-2DA1-DD11-AC0E-0019DB29C620.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3CB67850-2DA1-DD11-8A02-001617C3B5F4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/5EE0D76B-25A1-DD11-882D-001D09F23174.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2C32DD39-28A1-DD11-8988-001D09F28E80.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/CAB01506-16A1-DD11-9707-0030487C6090.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/905EFF3E-16A1-DD11-8D6C-000423D98750.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/54764E36-12A1-DD11-9DA3-00304879FBB2.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/6EBFCACD-0EA1-DD11-8B7A-000423D6BA18.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/062129E0-12A1-DD11-ABE3-001617C3B5D8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C00E5F67-1CA1-DD11-8ED6-000423D990CC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A4DCCCD5-15A1-DD11-8E1D-001D09F231B0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C6A35D2C-18A1-DD11-A5BF-001D09F253FC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/6A306057-1EA1-DD11-AF79-000423D98B5C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3C7C8EE4-17A1-DD11-8C38-0030487D1BCC.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/829A80D7-19A1-DD11-8CCF-001D09F24047.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/54BBF77B-24A1-DD11-B17A-0030487A322E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C68981B2-1EA1-DD11-B4ED-000423D94A20.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/14F510BF-1EA1-DD11-9604-000423D9870C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/80BB914B-1BA1-DD11-AA2D-001D09F24498.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/08E20AE6-1CA1-DD11-AB61-000423D6B358.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/72BD51D3-34A1-DD11-8345-000423D98C20.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/48CCC69A-2CA1-DD11-B259-001D09F24EC0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/84186420-31A1-DD11-97F2-001D09F2432B.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/44117B58-2EA1-DD11-8663-0030487A322E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3AC06681-2EA1-DD11-8282-000423D98DD4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/44E936E2-29A1-DD11-8F2F-000423D9853C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/44508136-2EA1-DD11-A681-000423D98E6C.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/1AC9913C-37A1-DD11-86C2-0019B9F7310E.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/5423A355-32A1-DD11-8627-001D09F23D1D.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/DA616F90-34A1-DD11-8B53-001D09F24934.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/66BBB1C6-26A1-DD11-8655-0019B9F72D71.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/166135C5-26A1-DD11-8CA9-000423D98DD4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2058E638-10A1-DD11-A22C-0030487A3C9A.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/D6E39B97-0FA1-DD11-84FD-000423D94AA8.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/62CAA0DC-1BA1-DD11-9D6E-0019B9F72F97.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/3A4AD6F8-1CA1-DD11-9C02-000423D985B0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/5EE573CF-16A1-DD11-AAC5-001D09F23E53.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/22E93A70-23A1-DD11-8FDC-000423D98834.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/2E545B40-2FA1-DD11-8970-001D09F25456.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/6C7D90CE-2CA1-DD11-A3FF-001D09F24600.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C815B7BC-28A1-DD11-BA12-001D09F252E9.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/A8E2DEF6-2DA1-DD11-B1CA-001D09F23D04.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/4258C491-2AA1-DD11-9548-001D09F25442.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/C0688CE0-2AA1-DD11-848C-001D09F252E9.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/00DD0655-37A1-DD11-BFCE-001D09F253D4.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/AADCAEC3-35A1-DD11-A826-001D09F241F0.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/9C9385BB-31A1-DD11-931E-001D09F244BB.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/62F77644-24A1-DD11-B441-001D09F28755.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/60AFA979-24A1-DD11-B40D-001D09F250AF.root",
"/store/data/Commissioning08/Cosmics/RAW/v1/000/067/173/142D6F19-24A1-DD11-A97D-001D09F24600.root",
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)



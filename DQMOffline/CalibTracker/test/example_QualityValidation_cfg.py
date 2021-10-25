import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripHotStripCalibration")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('orbitFilter','OrbitFilter','SiStripQualityHotStripIdentifierRoot','siStripQualityHotStripIdentifierRoot','SiStripHotStripAlgorithmFromClusterOccupancy'),
    log = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('log.txt')
)

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect='frontier://FrontierProd/CMS_COND_21X_STRIP'
process.poolDBESSource=cms.ESSource("PoolDBESSource",
                                    process.CondDBCommon,
                                    BlobStreamerName=cms.untracked.string('TBufferBlobStreamingService'),
                                    toGet           =cms.VPSet(
    cms.PSet(
    record=cms.string('SiStripBadModuleRcd'),
    tag   =cms.string('SiStripBadChannel_HotStrip_CRAFT_v1_offline')
    )
    )
                                    )

# Include masking #

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
 ,cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string(''))
 ,cms.PSet(record=cms.string('SiStripBadModuleRcd' ),tag=cms.string(''))
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)


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
#process.SiStripMonitorCluster.TH1ClusterStoNVsPos.moduleswitchon = cms.bool(True)

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
                            #firstRun = cms.untracked.uint32(67173),
                            #lastRun = cms.untracked.uint32(67173),
                            #skipEvents = cms.untracked.uint32(4000000),
                            interval = cms.uint32(1),
                            fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/0E017228-C2AA-DD11-99E3-000423D98804.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/103883BF-C2AA-DD11-A0E7-000423D9939C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/12A69907-C2AA-DD11-9A5B-000423D6B2D8.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/1C363A46-C6AA-DD11-A731-000423D60FF6.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/1E8D2250-C6AA-DD11-9387-000423D990CC.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/26DB7910-C2AA-DD11-8543-000423D6B48C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/2A3D3237-D2AA-DD11-9356-001617C3B69C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/3E4861A0-C5AA-DD11-B728-000423D9870C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/3EA1EDFF-C6AA-DD11-BFBE-001617C3B70E.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/4C92B0DB-C4AA-DD11-8AED-000423D99394.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/6211DBD5-C4AA-DD11-95A1-001617C3B77C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/625124F0-C6AA-DD11-8408-001617C3B706.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/702DD7DC-C4AA-DD11-BCA2-000423D9880C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/84AED28C-C5AA-DD11-98FD-000423D98950.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/88F6DE47-C6AA-DD11-B212-001617C3B6CC.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/8AD6A56F-C3AA-DD11-8D42-000423D94494.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/8CCE5DF6-C6AA-DD11-91DD-000423D98BC4.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/90723321-C4AA-DD11-91A4-001617DBD230.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/94F6A026-C4AA-DD11-883D-000423D98634.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/9AC3020D-C2AA-DD11-A7B6-000423D99614.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/9AD1D9BD-C2AA-DD11-82C6-000423D98FBC.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/B0E82E0A-C2AA-DD11-BA63-000423D98AF0.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/B8CD766F-C3AA-DD11-82C0-001617E30D00.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/BA76AF56-C1AA-DD11-A6AF-001617C3B77C.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/CE974773-C3AA-DD11-BDB7-000423D99996.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/D09AF0B6-C2AA-DD11-A3A5-000423D98920.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/E6FE1827-C4AA-DD11-8467-001617C3B73A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/EC3A6EA2-C5AA-DD11-A51E-001617C3B65A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/ECB45AB4-C0AA-DD11-B30B-000423D99264.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/F40D1BDA-C4AA-DD11-B764-001617DBCF6A.root',
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/351/FC9078C1-C2AA-DD11-8D83-000423D98634.root'
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



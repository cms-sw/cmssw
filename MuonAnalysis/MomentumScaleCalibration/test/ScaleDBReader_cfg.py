import FWCore.ParameterSet.Config as cms

process = cms.Process("SCALEDBREADER")
# process.load("MuonAnalysis.MomentumScaleCalibration.local_CSA08_Y_cff")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")

process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")

# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring()
# )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'),
    # connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_PHYSICSTOOLS'),
    # connect = cms.string('sqlite_file:MuScleFit_Scale_Z_36_invPb_innerTrack_Dec22_v1.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('MuScleFitDBobjectRcd'),
        # tag = cms.string('MuScleFit_Scale_Z_MC_Startup_innerTrack')
        # tag = cms.string('MuScleFit_Scale_Z_MC_Realistic2010_innerTrack')
        # tag = cms.string('MuScleFit_Scale_Z_36_invPb_innerTrack_Dec22_v1')
        tag = cms.string('MuScleFit_Scale_Jpsi_Data_2011_innerTrack')
    ))
)

process.DBReaderModule = cms.EDAnalyzer(
    "DBReader",
    # Specify that we want to write the scale parameters. THIS MUST NOT BE CHANGED.
    Type = cms.untracked.string("scale")
)

process.p1 = cms.Path(process.DBReaderModule)


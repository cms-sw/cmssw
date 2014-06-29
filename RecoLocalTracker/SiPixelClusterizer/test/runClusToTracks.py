# produce pixel cluster & rechits from digia
# works directly or through raw
# 
#
##############################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("CluToTrack")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# for data?
#process.load('Configuration.StandardSequences.Reconstruction_Data_cff')


# clusterizer 
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

# for raw
#process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
#process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')

# needed for pixel RecHits (templates?)
process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("RecoTracker.Configuration.RecoTracker_cff")

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('SiPixelClusterizer'),
    debugModules = cms.untracked.vstring('SiPixelRecHits'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
#       threshold = cms.untracked.string('INFO')
       threshold = cms.untracked.string('ERROR')
#        threshold = cms.untracked.string('WARNING')
#        threshold = cms.untracked.string('DEBUG')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)
# get the files from DBS:
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/tracks/tracks2_mc71.root'
#    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/tracks/tracks2_postls171.root'

 "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/F60495B3-1E41-E211-BB7C-003048D3756A.root",

    )
)

process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('208686:73-208686:463')
# Choose the global tag here:
# process.GlobalTag.globaltag = 'MC_71_V1::All'
# process.GlobalTag.globaltag = 'POSTLS171_V1::All'
# process.GlobalTag.globaltag = 'PRE_MC_71_V2::All'
# data 
#process.GlobalTag.globaltag = "GR_R_71_V1::All"
process.GlobalTag.globaltag = "PRE_R_71_V3::All"

#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(0),
#        authenticationPath = cms.untracked.string('')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiPixelQualityRcd'),
#        tag = cms.string('SiPixelBadModule_test')
#    )),
#    connect = cms.string('sqlite_file:test.db')
#)
#
# To use a test DB instead of the official pixel object DB tag: 
#process.customDead = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string('sqlite_file:/afs/cern.ch/user/v/vesna/Digitizer/dead_20100901.db'), toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityRcd'), tag = cms.string('dead_20100901'))))
#process.es_prefer_customDead = cms.ESPrefer("PoolDBESSource","customDead")


process.o1 = cms.OutputModule("PoolOutputModule",
#        outputCommands = cms.untracked.vstring('drop *','keep *_*_*_CluToTrack'),
        fileName = cms.untracked.string('file:tracks.root'),
#       fileName = cms.untracked.string('file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/tracks/tracks2_postls171.root'),
#    splitLevel = cms.untracked.int32(0),
#    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('RECO')
    )

)

# DB stuff 
useLocalDB = False
if useLocalDB :
# Frontier LA 
    process.DBReaderFrontier = cms.ESSource("PoolDBESSource",
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(0),
         authenticationPath = cms.untracked.string('')
     ),
     toGet = cms.VPSet(
 	 cms.PSet(
# GenError
          record = cms.string('SiPixelGenErrorDBObjectRcd'),
#          tag = cms.string('SiPixelGenErrorDBObject38Tv1')
#          tag = cms.string('SiPixelGenErrorDBObject38TV10')
#          tag = cms.string('SiPixelGenErrorDBObject38T_v0_mc1')
          tag = cms.string('SiPixelGenErrorDBObject_38T_v1_mc')
# LA
# 			record = cms.string("SiPixelLorentzAngleRcd"),
# 			label = cms.untracked.string("fromAlignment"),
# 			label = cms.untracked.string("forWidth"),
# 			tag = cms.string("SiPixelLorentzAngle_v02_mc")
# 			tag = cms.string("SiPixelLorentzAngle_fromAlignment_v0_mc")
# 			tag = cms.string("SiPixelLorentzAngle_forWidth_v0_mc")
 		),
# 		cms.PSet(
# 			record = cms.string("SiPixelLorentzAngleSimRcd"),
# 			tag = cms.string("test_LorentzAngle_Sim")
# 		)
 	),
#     connect = cms.string('frontier://FrontierProd/CMS_COND_31X_PIXEL')
     connect = cms.string('frontier://FrontierPrep/CMS_COND_PIXEL')
    ) # end process

# SQ_LITE GenError
    process.DBReaderFrontier2 = cms.ESSource("PoolDBESSource",
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(0),
         authenticationPath = cms.untracked.string('')
     ),
     toGet = cms.VPSet(
 		cms.PSet(
 			record = cms.string("SiPixelGenErrorDBObjectRcd"),
# 			label = cms.untracked.string("fromAlignment"),
# 			tag = cms.string("SiPixelGenErrorDBObject38Tv1")
 			tag = cms.string("SiPixelGenErrorDBObject38TV10")
 		),
 	),
#     connect = cms.string('sqlite_file:siPixelGenErrors38T.db_old')
     connect = cms.string('sqlite_file:siPixelGenErrors38T.db')
   ) # end process
# endif
 
#process.myprefer = cms.ESPrefer("PoolDBESSource","DBReaderFrontier")
#process.myprefer2 = cms.ESPrefer("PoolDBESSource","DBReaderFrontier2")


#process.Timing = cms.Service("Timing")
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

# My 
# modify clusterie parameters
#process.siPixelClusters.ClusterThreshold = 4000.0

# direct clusterization (no raw step)
# label of digis 
#process.siPixelClusters.src = 'mix'

# plus pixel clusters  (OK)
#process.p1 = cms.Path(process.siPixelClusters)

# plus pixel rechits (OK)
#process.p1 = cms.Path(process.pixeltrackerlocalreco)

# clusterize through raw (OK)
# for digi to raw
process.siPixelRawData.InputLabel = 'simSiPixelDigis'
process.SiStripDigiToRaw.InputModuleLabel = 'simSiPixelDigis'
# for Raw2digi for simulations 
process.siPixelDigis.InputLabel = 'siPixelRawData'
process.siStripDigis.ProductLabel = 'SiStripDigiToRaw'
# for digi to clu
process.siPixelClusters.src = 'siPixelDigis'

# pixel only 
#process.p1 = cms.Path(process.siPixelRawData)
#process.p1 = cms.Path(process.siPixelRawData*process.siPixelDigis)
#process.p1 = cms.Path(process.siPixelRawData*process.siPixelDigis*process.pixeltrackerlocalreco)

# with strips ok
#process.p1 = cms.Path(process.siPixelRawData*process.SiStripDigiToRaw)
#process.p1 = cms.Path(process.siPixelRawData*process.SiStripDigiToRaw*process.siPixelDigis*process.siStripDigis)

# runs ok
#process.p1 = cms.Path(process.trackerlocalreco)

# runs ok
#process.p1 = cms.Path(process.offlineBeamSpot)

# runs ok
#process.p1 = cms.Path(process.siPixelRawData*process.SiStripDigiToRaw*process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.MeasurementTrackerEvent)

# runs ok
#process.p1 = cms.Path(process.siPixelRawData*process.SiStripDigiToRaw*process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.MeasurementTrackerEvent*process.siPixelClusterShapeCache*process.recopixelvertexing)

process.d = cms.EDAnalyzer("TestWithTracks",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("generalTracks"),
#     PrimaryVertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),                             
#     trajectoryInput = cms.string("TrackRefitterP5")
#     trajectoryInput = cms.string('cosmictrackfinderP5')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo_tracks.root')
)


process.load("RecoTracker.IterativeTracking.iterativeTk_cff")

# copy the sequence below from  
# RecoTracker/IterativeTracking/python/iterativeTk_cff.py  - 71_pre7
process.myTracking = cms.Sequence(process.InitialStep*
                            process.DetachedTripletStep*
                            process.LowPtTripletStep*
                            process.PixelPairStep*
                            process.MixedTripletStep*
                            process.PixelLessStep*
                            process.TobTecStep*
                            process.earlyGeneralTracks*
                            # muonSeededStep*
                            process.preDuplicateMergingGeneralTracks*
                            process.generalTracksSequence*
                            process.ConvStep*
                            process.conversionStepTracks
                            )

# run full tracking
# trackingGlobalReco does not work, needs EarlyMuons for muon seeding.
# ckftracks & iterTracking does not work as well  (same problem).
process.p1 = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.offlineBeamSpot*process.siPixelClusterShapeCache*process.recopixelvertexing*process.MeasurementTrackerEvent*process.myTracking*process.vertexreco*process.d)

#process.p1 = cms.Path(process.siPixelRawData*process.SiStripDigiToRaw*process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.siPixelClusterShapeCache*process.recopixelvertexing*process.MeasurementTrackerEvent*process.myTracking*process.vertexreco)

#process.outpath = cms.EndPath(process.o1)

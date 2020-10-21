import FWCore.ParameterSet.Config as cms

process = cms.Process("DEDX")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(     threshold = cms.untracked.string('ERROR')    ),
    destinations = cms.untracked.vstring('cout')
)


#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'GR09_R_35X_V2::All'

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.TrajectoryInEvent = cms.bool(True)
#process.TrackRefitter.src = 'ALCARECOTkAlCosmicsCosmicTF0T'
process.TrackRefitter.src = 'ctfWithMaterialTracksP5'


process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(190000),
    lastValue  = cms.uint64(9999999)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)


process.dedxDiscrimMAP = cms.EDAnalyzer("DeDxDiscriminatorLearner",
#process.dedxDiscrimMAP = cms.EDAnalyzer("DeDxDiscriminatorLearnerFromCalibTree",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),
   
    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(False),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06),

    AlgoMode            = cms.untracked.string("WriteOnDB"),
    HistoFile           = cms.untracked.string("ProbaMap.root"),

    MaxNrStrips         = cms.untracked.uint32(255),

    P_Min               = cms.double  (1.0 ),
    P_Max               = cms.double  (15.0),
    P_NBins             = cms.int32   (14  ),
    Path_Min            = cms.double  (0.2 ),
    Path_Max            = cms.double  (1.6 ),
    Path_NBins          = cms.int32   (28  ),
    Charge_Min          = cms.double  (0   ),
    Charge_Max          = cms.double  (4000),
    Charge_NBins        = cms.int32   (400 ),

#    InputFiles         = cms.vstring(),


    SinceAppendMode     = cms.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripDeDxMip_3D_Rcd'),
    doStoreOnDB         = cms.bool(True)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:Data8TeV_Deco_SiStripDeDxMip_3D_Rcd.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDeDxMip_3D_Rcd'),
        tag = cms.string('Data8TeV_Deco_3D_Rcd_52X')
    ))
)



#process.p        = cms.Path(process.offlineBeamSpot + process.TrackRefitter + process.dedxDiscrimMAP)
process.p        = cms.Path(process.dedxDiscrimMAP)




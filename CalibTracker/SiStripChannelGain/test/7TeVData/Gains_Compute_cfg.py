import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')  ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(134721),
    lastValue  = cms.uint64(134721)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_35X_V8::All'
process.prefer("GlobalTag")

process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
process.SiStripCalib.InputFiles          = cms.vstring(
        'file:/storage/data/cms/store/user/quertenmont/10_05_03_CalibTree/Run135149/out_232_1.root',
        'file:/storage/data/cms/store/user/quertenmont/10_05_03_CalibTree/Run135149/out_169_1.root',
        'file:/storage/data/cms/store/user/quertenmont/10_05_03_CalibTree/Run135149/out_43_1.root',
        'file:/storage/data/cms/store/user/quertenmont/10_05_03_CalibTree/Run135149/out_173_1.root',
        'file:/storage/data/cms/store/user/quertenmont/10_05_03_CalibTree/Run135149/out_205_1.root',
        'file:/storage/data/cms/store/user/quertenmont/10_05_03_CalibTree/Run135149/out_41_1.root',
)



process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:Gains_Sqlite.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('IdealGainTag')
    ))
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Gains_Tree.root')  
)

process.p = cms.Path(process.SiStripCalib)

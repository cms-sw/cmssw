import FWCore.ParameterSet.Config as cms

process = cms.Process("analyze")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("DQMServices.Core.DQM_cfg")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load("CalibTracker.SiStripLorentzAngle.SiStripCalibLorentzAngle_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debug = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('write_langle_geom_ideal_blob')
)

process.TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(True)
)

process.Timing = cms.Service("Timing")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripBadModule_CRAFT_21X_byHandFlagFromRun69636_v1')
    ))
)

process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.sistripLACalib)
process.ep = cms.EndPath(process.print)
process.GlobalTag.globaltag = '1PB_V2_RECO::All'
process.DQM.collectorHost = ''
process.CondDBCommon.connect = 'sqlite_file:LA_CRAFT.db'
process.sistripLACalib.ModuleFit2ITXMin = -0.4
process.sistripLACalib.ModuleFit2ITXMax = 0.1
process.sistripLACalib.FitCuts_Entries = 1000
process.sistripLACalib.FitCuts_p0 = 0
process.sistripLACalib.FitCuts_p1 = 0.1
process.sistripLACalib.FitCuts_p2 = 1
process.sistripLACalib.FitCuts_chi2 = 30
process.sistripLACalib.fileName = 'Summary_new_bin.root'



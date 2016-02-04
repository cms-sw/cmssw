import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('dtTTrigOffsetCalibration')
process.MessageLogger.destinations = cms.untracked.vstring('cerr')
process.MessageLogger.categories.append('Calibration')
process.MessageLogger.cerr =  cms.untracked.PSet(
     threshold = cms.untracked.string('DEBUG'),
     noLineBreaks = cms.untracked.bool(False),
     DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
     INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
     Calibration = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_cff")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    #connect = cms.string('sqlite_file:ttrig_matchRPhi_67838.db'),
    connect = cms.string('sqlite_file:ttrig_t0SegCorr_67838.db'),
    authenticationMethod = cms.untracked.uint32(0)
)

process.es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','calibDB')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.dtTTrigOffsetCalibration = cms.EDAnalyzer("DTTTrigOffsetCalibration",
    # Label to retrieve 4D segments from the event
    recHits4DLabel = cms.InputTag('dt4DSegments'),
    # Switch for the check of noisy channels
    checkNoisyChannels = cms.bool(False),
    # The Maximum incident angle for Theta Seg (degrees)
    maxAngleZ = cms.double(999.),
    # The Maximum allowed chi2 for 4D Segment reconstruction 
    maxChi2 = cms.double(1000.0),
    # The Maximum incident angle for Phi Seg (degrees)
    maxAnglePhi = cms.double(999.),
    # Choose to calculate vDrift and t0 or just fill the TMax histograms
    doT0SegCorrection = cms.untracked.bool(False),
    # Name of the ROOT file which will contain the TMax histos
    rootFileName = cms.untracked.string('DTT0SegHistos.root'),
    # Choose the chamber you want to calibrate (default = "All"), specify the chosen chamber
    # in the format "wheel station sector" (i.e. "-1 3 10")
    calibChamber = cms.untracked.string('All')
)

process.p = cms.Path(process.reco*process.dtTTrigOffsetCalibration)



import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#this block is there to solve issue related to SiStripQualityRcd
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.load("CalibTracker.SiStripESProducers.fake.SiStripDetVOffFakeESSource_cfi")
process.es_prefer_fakeSiStripDetVOff = cms.ESPrefer("SiStripDetVOffFakeESSource","siStripDetVOffFakeESSource")


#process.MessageLogger = cms.Service("MessageLogger",
#    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')  ),
#    destinations = cms.untracked.vstring('cout')
#)



process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/FAFF2948-4EDF-E111-97FB-BCAEC518FF44.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/AA3A95D6-5ADF-E111-ACB3-0025901D629C.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/A6FD8EEA-4CDF-E111-AB2A-BCAEC5329700.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/A2358E47-4EDF-E111-A93D-BCAEC532971B.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/9A59F24F-57DF-E111-B724-5404A63886AE.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/9041A3ED-61DF-E111-B138-BCAEC5329713.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/8E28BE09-64DF-E111-B559-E0CB4E55365D.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/8A4AF852-4ADF-E111-B7D0-003048F024FE.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/629D881A-4FDF-E111-AA10-001D09F34488.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/3866F4D2-4ADF-E111-8749-5404A63886A0.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/287CFC74-59DF-E111-9175-5404A63886D4.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/26F372C9-5FDF-E111-9418-001D09F242EF.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/268097BA-58DF-E111-93A0-0025901D6288.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/2462FB44-4EDF-E111-8D3D-BCAEC518FF8A.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/1800C6E5-55DF-E111-AC90-003048F1C58C.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/16E003CB-4ADF-E111-8883-BCAEC518FF41.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/14F9B88E-56DF-E111-B79B-001D09F24303.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/0CC85F9E-88DF-E111-904F-001D09F29321.root','/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/00A42AC9-4ADF-E111-8749-5404A6388697.root',)
    )


process.GlobalTag.globaltag = 'GR_P_V40::All'

#process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
#process.SiStripCalib.InputFiles          = cms.vstring(
#XXX_CALIBTREE_XXX
#)
#process.SiStripCalib.FirstSetOfConstants = cms.untracked.bool(False)
#process.SiStripCalib.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module


process.SiStripCalib = cms.EDAnalyzer("SiStripGainFromCalibTree",
    OutputGains         = cms.string('Gains_ASCII.txt'),
    Tracks              = cms.untracked.InputTag('CalibrationTracksRefit'),
    AlgoMode            = cms.untracked.string('PCL'),

    #Gain quality cuts
    minNrEntries        = cms.untracked.double(25),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(25.0),

    #track/cluster quality cuts
    minTrackMomentum    = cms.untracked.double(2),
    maxNrStrips         = cms.untracked.uint32(8),

    Validation          = cms.untracked.bool(False),
    OldGainRemoving     = cms.untracked.bool(False),
    FirstSetOfConstants = cms.untracked.bool(True),

    CalibrationLevel    = cms.untracked.int32(0), # 0==APV, 1==Laser, 2==module

    InputFiles          = cms.vstring(),

    UseCalibration     = cms.untracked.bool(False),
    calibrationPath    = cms.untracked.string(""),

    SinceAppendMode     = cms.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(True),
)

process.SiStripCalib.FirstSetOfConstants = cms.untracked.bool(False)
process.SiStripCalib.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module

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





process.load('Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi')
process.load('RecoVertex.BeamSpotProducer.BeamSpot_cff')
process.load('RecoTracker.TrackProducer.TrackRefitters_cff')

process.CalibrationTracksRefit = process.TrackRefitter.clone(src = cms.InputTag("CalibrationTracks"))
process.CalibrationTracks = process.AlignmentTrackSelector.clone(
#    src = 'generalTracks',
    src = 'ALCARECOSiStripCalMinBias',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0.8,
    nHitMin = 6,
    chi2nMax = 10.,
    )

# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
process.trackFilterRefit = cms.Sequence( process.CalibrationTracks + process.offlineBeamSpot + process.CalibrationTracksRefit )

process.p = cms.Path(process.trackFilterRefit * process.SiStripCalib)

import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')  ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(165098),
    lastValue  = cms.uint64(165098)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_P_V20::All'
process.prefer("GlobalTag")

process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
process.SiStripCalib.InputFiles          = cms.vstring(
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165098.root",        #size = 63.6604MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165120.root",        #size = 184.067MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165121.root",        #size = 2403.3MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165205.root",        #size = 1226.66MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165208.root",        #size = 700.343MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165364.root",        #size = 4733.81MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165400.root",        #size = 310.319MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165402.root",        #size = 104.435MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165415.root",        #size = 5204.78MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165472.root",        #size = 2206.28MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165487.root",        #size = 157.755MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165506.root",        #size = 960.467MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165514.root",        #size = 5449.39MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165525.root",        #size = 283.49MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165537.root",        #size = 2434.62MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165542.root",        #size = 1411.93MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165548.root",        #size = 4380.3MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165558.root",        #size = 414.236MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165567.root",        #size = 5986.54MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165570.root",        #size = 8238.97MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165617.root",        #size = 2971.02MB
   "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/GR11//calibTree_165620.root",        #size = 532.071MB
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

process.p = cms.Path(process.SiStripCalib)

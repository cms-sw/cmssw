import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
#untracked PSet maxEvents = {untracked int32 input = 100}
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

#process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Configuration/StandardSequences/MagneticField_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#include "SimMuon/CSCDigitizer/data/muonCSCDbConditions.cfi"
#replace muonCSCDigis.stripConditions = "Database"
#replace muonCSCDigis.strips.ampGainSigma = 0.
#replace muonCSCDigis.strips.peakTimeSigma = 0.
#replace muonCSCDigis.strips.doNoise = false
#replace muonCSCDigis.wires.doNoise = false
#replace muonCSCDigis.strips.doCrosstalk = false
#process.load("CalibMuon.Configuration.CSC_FakeDBConditions_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# initialize MessageLogger and output report
process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('detailedInfo'),
  #destinations = cms.untracked.vstring('detailedInfo_754'),
  debugModules = cms.untracked.vstring('muonCSCDigis'),
  categories = cms.untracked.vstring(
    #'CSCDCCUnpacker|CSCRawToDigi', 'StatusDigis', 'StatusDigi', 'CSCRawToDigi', 'CSCDCCUnpacker', 'EventInfo',
    'badData'),
  detailedInfo = cms.untracked.PSet(
    INFO = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    TRACE = cms.untracked.PSet(limit = cms.untracked.int32(0) ),
    noTimeStamps = cms.untracked.bool(False),
    FwkReport = cms.untracked.PSet(
      reportEvery = cms.untracked.int32(1),
      limit = cms.untracked.int32(10000000)
    ),
    CSCRawToDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
    StatusDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
    EventInfo = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),

    default = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
    Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkSummary = cms.untracked.PSet(reportEvery = cms.untracked.int32(1), limit = cms.untracked.int32(10000000) ),
    threshold = cms.untracked.string('DEBUG')
  )
)

#   include
#   "CalibMuon/Configuration/data/CSC_FrontierConditions.cff"
#   replace
#   cscConditions.toGet
#   =
#   {
#        {
#        string
#        record
#        =
#        "CSCDBGainsRcd"
#          string
#          tag
#          =
#          "CSCDBGains_ideal"},
#        {string
#        record
#        =
#        "CSCNoiseMatrixRcd"
#          string
#          tag
#          =
#          "CSCNoiseMatrix_ideal"},
#        {string
#        record
#        =
#        "CSCcrosstalkRcd"
#          string
#          tag
#          =
#          "CSCCrosstalk_ideal"},
#        {string
#        record
#        =
#        "CSCPedestalsRcd"
#         string
#         tag
#         =
#         "CSCPedestals_ideal"}
#    }
#process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
#process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

#process.GlobalTag.globaltag = 'CRAFT_V3P::All'
process.GlobalTag.globaltag = "CRAFT_30X::All"

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
  debugFlag = cms.untracked.bool(True),
  #skipEvents = cms.untracked.uint32(18237),
  #skipEvents = cms.untracked.uint32(60),
  skipEvents = cms.untracked.uint32(0),
  #debugVebosity = cms.untracked.uint32(10),
  fileNames = cms.untracked.vstring(
    #'rfio:/afs/cern.ch/user/a/asakharo/scratch0/events/my_good_events.root'
    #'/store/data/Commissioning08/Cosmics/RAW/v1/000/064/257/A6B9F13F-CC90-DD11-9BCA-001617E30CE8.root'
    #'/store/data/Commissioning08/BeamHalo/RAW/GRtoBeam_v1/000/062/096/863014FF-7C7F-DD11-8E83-0019DB29C614.root'
    #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/365/84E8B55A-EEAA-DD11-A18C-001617C3B65A.root'
    #'rfio:/afs/cern.ch/user/a/asakharo/scratch0/events/run_66740_FED_errors.root'
    #'rfio:/castor/cern.ch/user/a/asakharo/CMSevents/run_66740_FED_errors.root'
    #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/422F78CA-7019-DE11-A599-001617E30CD4.root'
    '/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/422F78CA-7019-DE11-A599-001617E30CD4.root',
    '/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/764D08CA-7019-DE11-813F-001617C3B69C.root',
    '/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/963C5DCA-7019-DE11-9ABF-001617DBD316.root',
    '/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/C882B9D5-7219-DE11-8B69-000423D6BA18.root'
  )
)

process.DQMStore = cms.Service("DQMStore")

process.dump = cms.EDFilter("CSCDigiDump",
  wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
  empt = cms.InputTag(""),
  stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
  comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
)

process.muonCSCDigis = cms.EDFilter("CSCDCCUnpacker",
  PrintEventNumber = cms.untracked.bool(False),
  UseExaminer = cms.untracked.bool(True),
  ErrorMask = cms.untracked.uint32(0x0),
  InputObjects = cms.InputTag("rawDataCollector"),
  ExaminerMask = cms.untracked.uint32(0x1FEBF3F6),
  UnpackStatusDigis = cms.untracked.bool(True),
  UseSelectiveUnpacking = cms.untracked.bool(True),
  isMTCCData = cms.untracked.bool(False),
  Debug = cms.untracked.bool(True),
  VisualFEDInspect=cms.untracked.bool(True),
  VisualFEDShort=cms.untracked.bool(False)
)

process.out = cms.OutputModule("PoolOutputModule",
  dataset = cms.untracked.PSet(dataTier = cms.untracked.string('DIGI')),
  fileName = cms.untracked.string('digi_test.root'),
)

#process.d=cms.EDAnalyzer('EventContentAnalyzer')

#process.muonCSCDigis.InputObjects = "rawDataCollector"
#process.p = cms.Path(process.d)
process.p1 = cms.Path(process.muonCSCDigis)
#process.k = cms.Path(process.d)
#process.e = cms.EndPath(process.out)
#process.c=cms.Path(process.dump)

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
        FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        FwkSummary = cms.untracked.PSet(reportEvery = cms.untracked.int32(1), limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG')
    )
)

#   include "CalibMuon/Configuration/data/CSC_FrontierConditions.cff"
#   replace cscConditions.toGet =  {
#        { string record = "CSCDBGainsRcd"
#          string tag = "CSCDBGains_ideal"},
#        {string record = "CSCNoiseMatrixRcd"
#          string tag = "CSCNoiseMatrix_ideal"},
#        {string record = "CSCcrosstalkRcd"
#          string tag = "CSCCrosstalk_ideal"},
#        {string record = "CSCPedestalsRcd"
#         string tag = "CSCPedestals_ideal"}
#    }
#process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
#process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

process.GlobalTag.globaltag = 'CRAFT_V3P::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

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
   'rfio:/castor/cern.ch/user/a/asakharo/CMSevents/run_66740_FED_errors.root'
   #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/422F78CA-7019-DE11-A599-001617E30CD4.root'
    )
)


process.DQMStore = cms.Service("DQMStore")
process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")

process.muonCSCDigis = cms.EDProducer("CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool(False),
    # Use CSC examiner for corrupt or semi-corrupt data to avoid unpacker crashes
    UseExaminer = cms.untracked.bool(True),
    # This mask simply reduces error reporting
    ErrorMask = cms.untracked.uint32(0x0),
    # Define input to the unpacker
    #InputTag InputObjects = cscpacker:CSCRawData
    InputObjects = cms.InputTag("rawDataCollector"),
    # This mask is needed by the examiner if it's used
    ExaminerMask = cms.untracked.uint32(0x1FEBF3F6),
    #this flag disables unpacking of the Status Digis
    UnpackStatusDigis = cms.untracked.bool(True),
    # Use Examiner to unpack good chambers and skip only bad ones
    UseSelectiveUnpacking = cms.untracked.bool(True),
    #set this to true if unpacking MTCC data from summer-fall MTCC2006 
    isMTCCData = cms.untracked.bool(False),
    # turn on lots of output                            
    Debug = cms.untracked.bool(False),
    # Turn on the visual inspection of bad events
    VisualFEDInspect=cms.untracked.bool(True),
    VisualFEDShort=cms.untracked.bool(True)
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
#process.c=cms.Path(process.cscDigiDump)

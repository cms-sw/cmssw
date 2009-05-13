import FWCore.ParameterSet.Config as cms

process = cms.Process("ViewDigi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# initialize MessageLogger and output report
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    #destinations = cms.untracked.vstring('detailedInfo_754'),
    debugModules = cms.untracked.vstring('muonCSCDigis'),
    #categories = cms.untracked.vstring(
    #'CSCDCCUnpacker|CSCRawToDigi', 'StatusDigis', 'StatusDigi', 'CSCRawToDigi', 'CSCDCCUnpacker', 'EventInfo',
    #'badData'),
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

#process.GlobalTag.globaltag = 'CRAFT_V3P::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    #skipEvents = cms.untracked.uint32(18237),
    #skipEvents = cms.untracked.uint32(60),
    skipEvents = cms.untracked.uint32(1),
    #debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
   #'rfio:/afs/cern.ch/user/a/asakharo/scratch0/events/my_good_events.root'
   #'/store/data/Commissioning08/Cosmics/RAW/v1/000/064/257/A6B9F13F-CC90-DD11-9BCA-001617E30CE8.root'
   '/store/data/Commissioning08/BeamHalo/RAW/GRtoBeam_v1/000/062/096/863014FF-7C7F-DD11-8E83-0019DB29C614.root'
   #'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/365/84E8B55A-EEAA-DD11-A18C-001617C3B65A.root'
   #'rfio:/afs/cern.ch/user/a/asakharo/scratch0/events/run_66740_FED_errors.root'
   #'rfio:/castor/cern.ch/user/a/asakharo/CMSevents/run_66740_FED_errors.root'
   #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/422F78CA-7019-DE11-A599-001617E30CD4.root'
    )
)

process.muonCSCDigis = cms.EDFilter("CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool(False),
    # Use CSC examiner for corrupt or semi-corrupt data to avoid unpacker crashes
    UseExaminer = cms.bool(True),
    # This mask simply reduces error reporting
    ErrorMask = cms.uint32(0x0),
    # Define input to the unpacker
    #InputTag InputObjects = cscpacker:CSCRawData
    InputObjects = cms.InputTag("source"),
    # This mask is needed by the examiner if it's used
    ExaminerMask = cms.uint32(0x1FEBF3F6),
    #this flag disables unpacking of the Status Digis
    UnpackStatusDigis = cms.bool(False),
    # Use Examiner to unpack good chambers and skip only bad ones
    UseSelectiveUnpacking = cms.bool(True),
    #set this to true if unpacking MTCC data from summer-fall MTCC2006 
    isMTCCData = cms.untracked.bool(False),
    # turn on lots of output                            
    Debug = cms.untracked.bool(False),
    # Format Status Digi
    UseFormatStatus=cms.bool(False),
    # Suppress zero LCTs
    SuppressZeroLCT=cms.untracked.bool(False),
    # Turn on the visual inspection of bad events
    VisualFEDInspect=cms.untracked.bool(True),
    VisualFEDShort=cms.untracked.bool(True)
)

process.dump = cms.EDAnalyzer("CSCViewDigi",
               wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
	       stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
	       comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
	       rpcDigiTag = cms.InputTag("muonCSCDigis","MuonCSCRPCDigi"),
               alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
               clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
               corrclctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
	       statusDigiTag = cms.InputTag("muonCSCDigis","MuonCSCDCCFormatStatusDigi"),
               WiresDigiDump = cms.untracked.bool(True),
	       StripDigiDump = cms.untracked.bool(True),
	       ComparatorDigiDump = cms.untracked.bool(True),
	       RpcDigiDump = cms.untracked.bool(True),
               AlctDigiDump = cms.untracked.bool(True),
               ClctDigiDump = cms.untracked.bool(True),
               CorrClctDigiDump = cms.untracked.bool(True),
	       StatusDigiDump = cms.untracked.bool(False)
)

process.out = cms.OutputModule("PoolOutputModule",
                      dataset = cms.untracked.PSet(dataTier = cms.untracked.string('DIGI')),
                               fileName = cms.untracked.string('digi_test.root'),
                               )

process.EventContent=cms.EDAnalyzer('EventContentAnalyzer')

process.p1 = cms.Path(process.muonCSCDigis)
process.p2 = cms.Path(process.EventContent)
process.p3 = cms.Path(process.dump)
#process.p4 = cms.EndPath(process.out)
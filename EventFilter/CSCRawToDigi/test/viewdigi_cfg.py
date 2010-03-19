import FWCore.ParameterSet.Config as cms

process = cms.Process("ViewDigi")

# Dump of different types of digis produced 
# by CSC RawToDigi chane

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
    #destinations = cms.untracked.vstring('detailedInfo'),
    destinations = cms.untracked.vstring('cout'),
    #destinations = cms.untracked.vstring('detailedInfo_754'),
    debugModules = cms.untracked.vstring('muonCSCDigis'),
    #categories = cms.untracked.vstring(
    #'CSCDCCUnpacker|CSCRawToDigi', 'StatusDigis', 'StatusDigi', 'CSCRawToDigi', 'CSCDCCUnpacker', 'EventInfo',
    #'badData'),
    #detailedInfo = cms.untracked.PSet(
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        #TRACE = cms.untracked.PSet(limit = cms.untracked.int32(0) ),
        noTimeStamps = cms.untracked.bool(False),
        #FwkReport = cms.untracked.PSet(
        #    reportEvery = cms.untracked.int32(1),
        #    limit = cms.untracked.int32(10000000)
        #),
        #CSCRawToDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        #StatusDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        #EventInfo = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),

        default = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
        #Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        #FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        #FwkSummary = cms.untracked.PSet(reportEvery = cms.untracked.int32(1), limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG')
    )
)

#process.GlobalTag.globaltag = 'CRAFT_V3P::All'
process.GlobalTag.globaltag ='GR09_P_V4::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    #skipEvents = cms.untracked.uint32(18237),
    #skipEvents = cms.untracked.uint32(60),
    skipEvents = cms.untracked.uint32(695),
    #debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
   # 'rfio:/tmp/asakharo/FE5D634A-6ACE-DE11-8CB8-0030487A18F2.root'
   '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/120/331/FE5D634A-6ACE-DE11-8CB8-0030487A18F2.root'
    )
)

process.muonCSCDigis.UnpackStatusDigis=cms.bool(True)
process.muonCSCDigis.SuppressZeroLCT=cms.untracked.bool(True)


process.load("EventFilter.CSCRawToDigi.veiwDigi_cfi")

process.dumpCSCdigi.WiresDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.StripDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.ComparatorDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.RpcDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.AlctDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.ClctDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.CorrClctDigiDump = cms.untracked.bool(True)
process.dumpCSCdigi.StatusCFEBDump = cms.untracked.bool(True)
process.dumpCSCdigi.StatusDigiDump = cms.untracked.bool(True)

process.out = cms.OutputModule("PoolOutputModule",
                      dataset = cms.untracked.PSet(dataTier = cms.untracked.string('DIGI')),
                               fileName = cms.untracked.string('digi_test.root'),
                               )

process.EventContent=cms.EDAnalyzer('EventContentAnalyzer')

process.p1 = cms.Path(process.muonCSCDigis)
#process.p2 = cms.Path(process.EventContent)
process.p3 = cms.Path(process.dumpCSCdigi)
#process.p4 = cms.EndPath(process.out)
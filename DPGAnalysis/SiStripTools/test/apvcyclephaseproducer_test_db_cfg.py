import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("APVCyclePhaseProducerTestDBfile")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
#options.globalTag = "DONOTEXIST::All"

options.parseArguments()

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.files.infos = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(10000)
    )
)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(
        "/store/data/Run2012A/Jet/RAW/v1/000/191/226/0AFA4854-0986-E111-881D-5404A640A639.root",
        "/store/data/Run2012A/SingleElectron/RAW/v1/000/193/621/B2087C58-9F98-E111-90BD-002481E0D646.root",
        "/store/data/Run2012A/SingleElectron/RAW/v1/000/193/648/9AEA9A92-F498-E111-98EB-001D09F23A20.root",
        "/store/data/Run2012C/MinimumBias/RAW/v1/000/199/812/007E76FB-9ED8-E111-8986-003048D2BE12.root",
        "/store/data/Run2012C/MinimumBias2/RAW/v1/000/198/603/346E0D4C-6FCA-E111-AE04-002481E0D958.root",
        "/store/data/Run2012C/MinimumBias/RAW/v1/000/203/002/50D2F56A-5700-E211-9636-5404A63886AE.root",
        "/store/data/Run2012D/MinimumBias/RAW/v1/000/207/454/0428CA7D-B830-E211-8481-485B3977172C.root",
        "/store/data/Commissioning2013/Cosmics/RAW/v1/00000/74845390-8D47-E311-9701-003048CFB390.root"
#        "/store/data/Commissioning2013/Cosmics/RAW/v1/00000/007C8D94-9A46-E311-9596-003048F0E594.root",
#        "/store/data/Commissioning2013/Cosmics/RAW/v1/00000/06631F08-9A45-E311-9BFD-003048F17496.root"
),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

process.source.fileNames = cms.untracked.vstring(options.inputFiles)

#process.source = cms.Source("EmptySource",
#                            firstRun = cms.untracked.uint32(216322),
#                            numberEventsInRun = cms.untracked.uint32(10)
#                            )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
#process.GlobalTag.toGet = cms.VPSet(
#cms.PSet(record = cms.string("SiStripConfObjectRcd"),
#         label = cms.untracked.string("apvphaseoffsets"), #guess
#         tag = cms.string("SiStripAPVPhaseOffsets_v2_hlt"),
#         connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_STRIP")
#        )
#)

#-------------------------------------------------------------------------

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_withdefault_cff")
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")

#process.APVPhases.ignoreDB = cms.untracked.bool(True)

process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('apvcyclephaseproducer_test_db.root')
                                   )

process.p0 = cms.Path(process.scalersRawToDigi + process.APVPhases +
                      process.apvcyclephasemonitor )


import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = ""
#process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#High PU Run
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FEAA124F-7EF3-E011-9208-003048673374.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FE03EB11-5DF3-E011-B717-003048F117EC.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FAB7E40D-76F3-E011-8566-BCAEC53296FF.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FA1E6192-81F3-E011-B876-BCAEC53296FF.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FA085011-7FF3-E011-9DE3-003048D2BC4C.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F8A8B8E4-86F3-E011-BC5A-BCAEC53296F7.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F8A32D30-80F3-E011-9CCE-001D09F24489.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F86797FF-7DF3-E011-AC73-003048F024FE.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F6C69A11-7FF3-E011-A256-001D09F24EAC.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F62BF6A4-7BF3-E011-8A47-E0CB4E4408E3.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F491E355-60F3-E011-9860-0030486780E6.root"

    )
)
# Event Setup
##############
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")
process.GlobalTag.globaltag ='GR_R_43_V3::All'

#  LCT Printer
process.lctPrinter = cms.EDAnalyzer("LCTPrinter",
  WantSimL1Tracks = cms.untracked.bool(False)
)

#This path is for Real Data
process.p = cms.Path(process.csctfunpacker*process.lctPrinter)

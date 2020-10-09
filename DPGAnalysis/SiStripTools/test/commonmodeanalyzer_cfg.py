import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("CommonModeAnalyzer")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.placeholder = cms.untracked.bool(False)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

process.MessageLogger.debugModules=cms.untracked.vstring("commonmodeanalyzer")
#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")

process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)

process.seqProducers = cms.Sequence(process.seqEventHistoryReco)

process.load("DPGAnalysis.SiStripTools.commonmodeanalyzer_cfi")
process.commonmodeanalyzer.historyProduct = cms.InputTag("froml1abcHEs")
process.commonmodeanalyzer.selections = cms.VPSet(
        cms.PSet(label=cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
        cms.PSet(label=cms.string("TEC"),selection=cms.untracked.vstring("0x1e000000-0x1c000000")),
        cms.PSet(label=cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),
        cms.PSet(label=cms.string("TID"),selection=cms.untracked.vstring("0x1e000000-0x18000000")),
        cms.PSet(label=cms.string("TOB_L1"),selection=cms.untracked.vstring("0x1e01c000-0x1a004000")),
        cms.PSet(label=cms.string("TOB_L2"),selection=cms.untracked.vstring("0x1e01c000-0x1a008000")),
        cms.PSet(label=cms.string("TOBplus_1_4_1_4"),selection=cms.untracked.vstring("0x1e01ffe0-0x1a006460")),
        cms.PSet(label=cms.string("TOBplus_2_2_2_4"),selection=cms.untracked.vstring("0x1e01ffe0-0x1a00a280"))
)
process.commonmodenoignorebadfedmod = process.commonmodeanalyzer.clone(ignoreBadFEDMod=False)
process.commonmodenoignore = process.commonmodenoignorebadfedmod.clone(ignoreNotConnected=False)

process.siStripDigis.UnpackCommonModeValues=cms.bool(True)

process.p0 = cms.Path(
   process.siStripDigis + process.siStripZeroSuppression +
   process.scalersRawToDigi +
   process.seqProducers +
   process.seqEventHistory +
   process.commonmodeanalyzer +
   process.commonmodenoignorebadfedmod + 
   process.commonmodenoignore 
   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('CommonModeAnalyzer.root')
                                   )

#print process.dumpPython()

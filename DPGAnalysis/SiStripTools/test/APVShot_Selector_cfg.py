import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("APVShotAnalyzer")

#prepare options

process.DQMStore=cms.Service("DQMStore")
process.TkDetMap=cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")


process.MessageLogger.debugModules=cms.untracked.vstring("apvshotfilter")
#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")

process.seqEventHistoryReco = process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)

process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")


process.load("DPGAnalysis.SiStripTools.apvshotsfilter_cfi")
process.apvshotsfilter.useCabling = cms.untracked.bool(True)

process.load("DPGAnalysis.SiStripTools.apvshotsanalyzer_cfi")
process.apvshotsanalyzer.historyProduct = cms.InputTag("froml1abcHEs")
process.apvshotsanalyzer.useCabling = cms.untracked.bool(True)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')



process.p0 = cms.Path(
   process.siStripDigis + process.siStripZeroSuppression +
   process.scalersRawToDigi +
   process.seqEventHistoryReco +
   process.seqEventHistory +
   process.apvshotsfilter +
   process.apvshotsanalyzer
   )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('APVShotAnalyzer.root')
                                   )


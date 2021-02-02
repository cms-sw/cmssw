import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("APVShotAnalyzer")

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

process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.enable = cms.untracked.bool(True)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

process.MessageLogger.debugModules=cms.untracked.vstring("apvshotsanalyzer")
#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

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

process.eventtimedistrFH1 = process.eventtimedistribution.clone()
process.eventtimedistrFH2 = process.eventtimedistribution.clone()

process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)
process.seqEventHistoryFH1 = cms.Sequence(process.eventtimedistrFH1)
process.seqEventHistoryFH2 = cms.Sequence(process.eventtimedistrFH2)

process.seqProducers = cms.Sequence(process.seqEventHistoryReco)

process.load("DPGAnalysis.SiStripTools.apvshotsanalyzer_cfi")
process.apvshotsanalyzer.historyProduct = cms.InputTag("froml1abcHEs")
process.apvshotsanalyzer.useCabling = cms.untracked.bool(True)
process.apvshotsanalyzerFH1 = process.apvshotsanalyzer.clone()
process.apvshotsanalyzerFH2 = process.apvshotsanalyzer.clone()

process.FrameHeader1Events = cms.EDFilter('EventWithHistoryEDFilter',
                                          commonConfiguration = cms.untracked.PSet(
                                                                                   historyProduct = cms.untracked.InputTag("froml1abcHEs"),
                                                                                   APVPhaseLabel = cms.untracked.string("APVPhases"),
                                                                                   partitionName = cms.untracked.string("Any")
                                                                                    ),
                                          filterConfigurations = cms.untracked.VPSet(
                                                                                     cms.PSet(
                                                                                              dbxInCycleRangeLtcyAware = cms.untracked.vint32(298,300)
                                                                                              )
                                                                                     )
                                          )
process.FrameHeader2Events = cms.EDFilter('EventWithHistoryEDFilter',
                                          commonConfiguration = cms.untracked.PSet(
                                                                                   historyProduct = cms.untracked.InputTag("froml1abcHEs"),
                                                                                   APVPhaseLabel = cms.untracked.string("APVPhases"),
                                                                                   partitionName = cms.untracked.string("Any")
                                                                                    ),
                                          filterConfigurations = cms.untracked.VPSet(
                                                                                     cms.PSet(
                                                                                              dbxInCycleRangeLtcyAware = cms.untracked.vint32(301,303)
                                                                                              )
                                                                                     )
                                          )


process.load("DQM.SiStripCommon.TkHistoMap_cff")

process.p0 = cms.Path(
   process.siStripDigis + process.siStripZeroSuppression +
   process.scalersRawToDigi +
   process.seqProducers +
   process.seqEventHistory +
   process.apvshotsanalyzer
   )
process.pfh1 = cms.Path(
   process.siStripDigis + process.siStripZeroSuppression +
   process.scalersRawToDigi +
   process.seqProducers +
   process.FrameHeader1Events +
   process.seqEventHistoryFH1 +
   process.apvshotsanalyzerFH1
   )
process.pfh2 = cms.Path(
   process.siStripDigis + process.siStripZeroSuppression +
   process.scalersRawToDigi +
   process.seqProducers +
   process.FrameHeader2Events +
   process.seqEventHistoryFH2 +
   process.apvshotsanalyzerFH2
   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('APVShotAnalyzer.root')
                                   )

#print process.dumpPython()

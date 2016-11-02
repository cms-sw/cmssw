import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("TIDTECInnerRingInvestigator")

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


#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------


process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")

process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)

process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")



process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")

#process.ssclustermultprod.withClusterSize=cms.untracked.bool(True)

process.ssclustermultprod.wantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("TIDring1"),selection=cms.untracked.vstring("0x1e000600-0x18000200")),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("TIDring2"),selection=cms.untracked.vstring("0x1e000600-0x18000400")),
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("TECring1"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000020")),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("TECring2"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000040"))
)

process.seqMultProd = cms.Sequence(process.ssclustermultprod)


process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterToDigiProducer_cfi")


process.seqProducers = cms.Sequence(process.seqEventHistoryReco + process.seqMultProd + process.siStripClustersToDigis)

#from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
#process.hltSelection = triggerResultsFilter.clone(
#                                          triggerConditions = cms.vstring("HLT_ZeroBias_*"),
#                                          hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
#                                          l1tResults = cms.InputTag( "" ),
#                                          throw = cms.bool(False)
#                                          )


process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
process.ssclusmultinvestigator.runHisto = cms.untracked.bool(True)
process.ssclusmultinvestigator.scaleFactor=cms.untracked.int32(1)
process.ssclusmultinvestigator.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("TIDring1"), binMax = cms.int32(1000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("TIDring2"), binMax = cms.int32(1000)),
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("TECring1"), binMax = cms.int32(1000)),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("TECring2"), binMax = cms.int32(1000))
    )

process.load("DPGAnalysis.SiStripTools.clusterbigeventsdebugger_cfi")
process.clusterbigeventsdebugger.selections = cms.VPSet(
cms.PSet(detSelection = cms.uint32(101),label = cms.string("TIDring1"),selection=cms.untracked.vstring("0x1e000600-0x18000200")),
cms.PSet(detSelection = cms.uint32(102),label = cms.string("TIDring2"),selection=cms.untracked.vstring("0x1e000600-0x18000400")),
cms.PSet(detSelection = cms.uint32(201),label = cms.string("TECring1"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000020")),
cms.PSet(detSelection = cms.uint32(202),label = cms.string("TECring2"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000040"))
)

process.load("DPGAnalysis.SiStripTools.digibigeventsdebugger_cfi")
process.digibigeventsdebugger.selections = process.clusterbigeventsdebugger.selections
process.digibigeventsdebugger.collection = cms.InputTag("siStripClustersToDigis","ZeroSuppressed")
#process.digibigeventsdebugger.foldedStrips = cms.untracked.bool(True)

process.seqClusMultInvest = cms.Sequence(process.ssclusmultinvestigator + process.clusterbigeventsdebugger + process.digibigeventsdebugger ) 



process.p0 = cms.Path(
    #    process.hltSelection +
    process.seqProducers +
    process.seqEventHistory +
    process.seqClusMultInvest)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('TIDTECInnerRingInvestigator_'+options.tag+'.root')
                                   )

#print process.dumpPython()

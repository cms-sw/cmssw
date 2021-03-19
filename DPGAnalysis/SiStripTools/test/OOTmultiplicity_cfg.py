import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

process = cms.Process("OOTmultiplicity")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.parseArguments()

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

#----------------------------------------------------------------


process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.CondDBESSource = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
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

process.MessageLogger.suppressWarning.append("consecutiveHEs")


#------------------------------------------------------------------

#process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.debugModules = cms.untracked.vstring("eventtimedistribution")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )



process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi 
process.APVPhases = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi.APVPhases 

process.APVPhases.wantHistos = cms.untracked.bool(True)

# Filter for no noisy bins

process.load("DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_AlCaReco_cfi")
process.load("DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_AlCaReco_cfi")
process.load("DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_additionalpeak_AlCaReco_cfi")

process.latencyPlusOne = cms.EDFilter('EventWithHistoryEDFilter',
                                      commonConfiguration = cms.untracked.PSet(historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                                                                               APVPhaseLabel  = cms.untracked.string("APVPhases"),
                                                                               ),
                                      filterConfigurations = cms.untracked.VPSet(cms.PSet(apvModes = cms.untracked.vint32(47),
                                                                                          dbxRangeLtcyAware = cms.untracked.vint32(1,1)
                                                                                          )
                                                                                 )
                                      )

process.PotentialNewHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                              commonConfiguration = cms.untracked.PSet(historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                                                                                       APVPhaseLabel  = cms.untracked.string("APVPhases"),
                                                                                       ),
                                              filterConfigurations = cms.untracked.VPSet(cms.PSet(apvModes = cms.untracked.vint32(37),
                                                                                                  partitionName = cms.untracked.string("Any"),
                                                                                                  absBXInCycleRangeLtcyAware = cms.untracked.vint32(12,12)
                                                                                                  )
                                                                                         )
                                              )

process.PotentialAdditionalHugeEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                                     commonConfiguration = cms.untracked.PSet(historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                                                                                              APVPhaseLabel  = cms.untracked.string("APVPhases"),
                                                                                              ),
                                                     filterConfigurations = cms.untracked.VPSet(cms.PSet(apvModes = cms.untracked.vint32(47),
                                                                                                  partitionName = cms.untracked.string("Any"),
                                                                                                  absBXInCycleRangeLtcyAware = cms.untracked.vint32(9,10)
                                                                                                  )
                                                                                         )
                                              )


process.seqNoNoisyBins = cms.Sequence(~process.PotentialTIBTECHugeEvents
                                      + ~process.PotentialTIBTECFrameHeaderEventsFPeak
                                      + ~process.PotentialTIBTECFrameHeaderEventsAdditionalPeak
                                      + ~process.latencyPlusOne
                                      + ~process.PotentialNewHugeEvents
                                      + ~process.PotentialAdditionalHugeEvents
                                      )


process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.wantEWHDepthHisto = cms.untracked.bool(True)
process.eventtimedistrRandom = process.eventtimedistribution.clone()
process.eventtimedistrnonoisybins = process.eventtimedistribution.clone()
process.eventtimedistrnonoisybinsRandom = process.eventtimedistribution.clone()


# cluster analysis

process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")
process.ssclustermultprod.wantedSubDets.extend(cms.VPSet(
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("TIBL1"),selection=cms.untracked.vstring("0x1e01c000-0x16004000")),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("TIBL2"),selection=cms.untracked.vstring("0x1e01c000-0x16008000")),
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIBL3"),selection=cms.untracked.vstring("0x1e01c000-0x1600c000")),
    cms.PSet(detSelection = cms.uint32(104),detLabel = cms.string("TIBL4"),selection=cms.untracked.vstring("0x1e01c000-0x16010000")),
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("TOBL1"),selection=cms.untracked.vstring("0x1e01c000-0x1a004000")),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("TOBL2"),selection=cms.untracked.vstring("0x1e01c000-0x1a008000")),
    cms.PSet(detSelection = cms.uint32(203),detLabel = cms.string("TOBL3"),selection=cms.untracked.vstring("0x1e01c000-0x1a00c000")),
    cms.PSet(detSelection = cms.uint32(204),detLabel = cms.string("TOBL4"),selection=cms.untracked.vstring("0x1e01c000-0x1a010000")),
    cms.PSet(detSelection = cms.uint32(205),detLabel = cms.string("TOBL5"),selection=cms.untracked.vstring("0x1e01c000-0x1a014000")),
    cms.PSet(detSelection = cms.uint32(206),detLabel = cms.string("TOBL6"),selection=cms.untracked.vstring("0x1e01c000-0x1a018000"))
))

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")

process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
process.ssclusmultinvestigator.scaleFactor = cms.untracked.int32(2)
process.ssclusmultinvestigator.fillHisto = cms.untracked.bool(True)
process.ssclusmultinvestigator.wantedSubDets.extend(cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("TIBL1"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("TIBL2"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIBL3"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(104),detLabel = cms.string("TIBL4"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("TOBL1"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("TOBL2"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(203),detLabel = cms.string("TOBL3"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(204),detLabel = cms.string("TOBL4"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(205),detLabel = cms.string("TOBL5"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(206),detLabel = cms.string("TOBL6"),  binMax = cms.int32(10000))
))
process.ssclusmultinvestRandom = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestnonoisybins = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestnonoisybinsRandom = process.ssclusmultinvestigator.clone()

process.load("DPGAnalysis.SiStripTools.spclusmultinvestigator_cfi")
process.spclusmultinvestigator.scaleFactor = cms.untracked.int32(2)
process.spclusmultinvestigator.fillHisto = cms.untracked.bool(True)
process.spclusmultinvestRandom = process.spclusmultinvestigator.clone()
process.spclusmultinvestnonoisybins = process.spclusmultinvestigator.clone()
process.spclusmultinvestnonoisybinsRandom = process.spclusmultinvestigator.clone()

process.load("DPGAnalysis.SiStripTools.ssclusmulttimecorrelations_cfi")
process.ssclusmulttimecorrelations.historyProduct = cms.InputTag("consecutiveHEs")
process.ssclusmulttimecorrelations.scaleFactors = cms.untracked.vint32()
process.ssclusmulttimecorrelations.runHisto = cms.untracked.bool(True)
process.ssclusmulttimecorrelations.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64)),
    cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/64)),
    cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID"), binMax = cms.int32( 565248/64)),
    cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB"), binMax = cms.int32(3303936/64)),
    cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"), binMax = cms.int32(3866624/64)),
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("TIBL1"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("TIBL2"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIBL3"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(104),detLabel = cms.string("TIBL4"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("TOBL1"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("TOBL2"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(203),detLabel = cms.string("TOBL3"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(204),detLabel = cms.string("TOBL4"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(205),detLabel = cms.string("TOBL5"),  binMax = cms.int32(10000)),
    cms.PSet(detSelection = cms.uint32(206),detLabel = cms.string("TOBL6"),  binMax = cms.int32(10000))
)
process.ssclusmulttimecorrRandom = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrnonoisybins = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrnonoisybinsRandom = process.ssclusmulttimecorrelations.clone()

process.load("DPGAnalysis.SiStripTools.spclusmulttimecorrelations_cfi")
process.spclusmulttimecorrelations.historyProduct = cms.InputTag("consecutiveHEs")
process.spclusmulttimecorrelations.scaleFactors = cms.untracked.vint32()
process.spclusmulttimecorrelations.runHisto = cms.untracked.bool(True)
process.spclusmulttimecorrRandom = process.spclusmulttimecorrelations.clone()
process.spclusmulttimecorrnonoisybins = process.spclusmulttimecorrelations.clone()
process.spclusmulttimecorrnonoisybinsRandom = process.spclusmulttimecorrelations.clone()

#
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.hltZeroBias = triggerResultsFilter.clone(
    triggerConditions = cms.vstring("HLT_ZeroBias_*"),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool(False)
                                          )
process.hltRandom = triggerResultsFilter.clone(
    triggerConditions = cms.vstring("HLT_Random_*"),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool(False)
    )



process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('OOTmultiplicity.root')
                                   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.pzerobias = cms.Path(process.hltZeroBias
                             + process.consecutiveHEs
                             + process.APVPhases
                             + process.apvcyclephasemonitor
                             + process.eventtimedistribution
                             + process.ssclustermultprod + process.ssclusmultinvestigator
#                             + process.ssclusmulttimecorrelations 
                             + process.spclustermultprod + process.spclusmultinvestigator
#                             + process.spclusmulttimecorrelations
                             )

process.prandom = cms.Path(process.hltRandom
                           + process.consecutiveHEs
                           + process.APVPhases
                           + process.apvcyclephasemonitor
                           + process.eventtimedistrRandom
                           + process.ssclustermultprod + process.ssclusmultinvestRandom
#                           + process.ssclusmulttimecorrRandom 
                           + process.spclustermultprod + process.spclusmultinvestRandom
#                           + process.spclusmulttimecorrRandom
                           )


process.pnonoisybins = cms.Path(process.hltZeroBias
                                + process.consecutiveHEs
                                + process.APVPhases
                                + process.seqNoNoisyBins
                                + process.ssclustermultprod 
                                + process.spclustermultprod 
                                + process.eventtimedistrnonoisybins
                                + process.ssclusmultinvestnonoisybins
#                                + process.ssclusmulttimecorrnonoisybins
                                + process.spclusmultinvestnonoisybins
#                                + process.spclusmulttimecorrnonoisybins
                                )

process.pnonoisybinsRandom = cms.Path(process.hltRandom
                                      + process.consecutiveHEs
                                      + process.APVPhases
                                      + process.seqNoNoisyBins
                                      + process.ssclustermultprod 
                                      + process.spclustermultprod 
                                      + process.eventtimedistrnonoisybinsRandom
                                      + process.ssclusmultinvestnonoisybinsRandom
#                                      + process.ssclusmulttimecorrnonoisybinsRandom
                                      + process.spclusmultinvestnonoisybinsRandom
#                                      + process.spclusmulttimecorrnonoisybinsRandom
                                )


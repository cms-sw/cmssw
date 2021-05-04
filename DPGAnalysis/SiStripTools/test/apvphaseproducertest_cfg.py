import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

process = cms.Process("apvphaseTest")

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

#----Remove too verbose PrimaryVertexProducer

process.MessageLogger.suppressInfo.append("pixelVerticesAdaptive")
process.MessageLogger.suppressInfo.append("pixelVerticesAdaptiveNoBS")

#----Remove too verbose BeamSpotOnlineProducer

process.MessageLogger.suppressInfo.append("testBeamSpot")
process.MessageLogger.suppressInfo.append("onlineBeamSpot")
process.MessageLogger.suppressWarning.append("testBeamSpot")
process.MessageLogger.suppressWarning.append("onlineBeamSpot")

#----Remove too verbose TrackRefitter

process.MessageLogger.suppressInfo.append("newTracksFromV0")
process.MessageLogger.suppressInfo.append("newTracksFromOtobV0")

#process.MessageLogger.suppressWarning.append("consecutiveHEs")
process.MessageLogger.suppressInfo.append("l1abcdebugger")

#------------------------------------------------------------------

#process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.debugModules = cms.untracked.vstring("eventtimedistribution")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )



fileNames = cms.untracked.vstring('')

process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")
process.load("DPGAnalysis.SiStripTools.l1abcdebugger_cfi")
process.l1abcdebuanydcs = process.l1abcdebugger.clone()

#process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts_cfi")

import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi 
process.APVPhases = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi.APVPhases 
#import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2013_cfi 
#process.APVPhases = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2013_cfi.APVPhases 
#process.APVPhases.defaultPhases = cms.vint32(57,57,57,57)
#process.APVPhases.magicOffset = cms.untracked.int32(8)

#import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2010_cfi 
#process.APVPhases2010 = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2010_cfi.APVPhases 

#process.APVPhasesEC02010 = process.APVPhases2010.clone(useEC0 = cms.untracked.bool(True),magicOffset = cms.untracked.int32(0))
#process.APVPhasesEC02011 = process.APVPhases2011.clone(useEC0 = cms.untracked.bool(True),magicOffset = cms.untracked.int32(0))

process.load("DPGAnalysis.SiStripTools.l1TSDebugger_cfi")

#------- Filter for isolated pairs
process.isoPairs5000 = cms.EDFilter('EventWithHistoryEDFilter',
                            commonConfiguration = cms.untracked.PSet(
                                  historyProduct             = cms.untracked.InputTag("consecutiveHEs"),
                                  APVPhaseLabel              = cms.untracked.string("APVPhases")
                                  ),
                            filterConfigurations = cms.untracked.VPSet(
                                  cms.PSet(
#                                          dbxTripletRange = cms.untracked.vint32(5000,-1)
                                          dbxGenericRange = cms.untracked.vint32(5000,-1),
                                          dbxGenericFirst = cms.untracked.uint32(1),dbxGenericLast = cms.untracked.uint32(2)
                                          )
                                  )
                            )
process.isoTriplets5000 = cms.EDFilter('EventWithHistoryEDFilter',
                            commonConfiguration = cms.untracked.PSet(
                                  historyProduct             = cms.untracked.InputTag("consecutiveHEs"),
                                  APVPhaseLabel              = cms.untracked.string("APVPhases")
                                  ),
                            filterConfigurations = cms.untracked.VPSet(
                                  cms.PSet(
#                                          dbxTripletRange = cms.untracked.vint32(5000,-1)
                                          dbxGenericRange = cms.untracked.vint32(5000,-1),
                                          dbxGenericFirst = cms.untracked.uint32(2),dbxGenericLast = cms.untracked.uint32(3)
                                          )
                                  )
                            )

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


# ------------------ Filters for peak and deco

process.DecoEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                  commonConfiguration = cms.untracked.PSet(historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                                                                           APVPhaseLabel  = cms.untracked.string("APVPhases"),
                                                                           ),
                                  filterConfigurations = cms.untracked.VPSet(cms.PSet(apvModes = cms.untracked.vint32(37))
                                                                             )
                                  )
process.PeakEvents = cms.EDFilter('EventWithHistoryEDFilter',
                                  commonConfiguration = cms.untracked.PSet(historyProduct = cms.untracked.InputTag("consecutiveHEs"),
                                                                           APVPhaseLabel  = cms.untracked.string("APVPhases"),
                                                                           ),
                                  filterConfigurations = cms.untracked.VPSet(cms.PSet(apvModes = cms.untracked.vint32(47))
                                                                             )
                                  )


process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")
#process.apvcyclephasemonitorEC02011 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhasesEC02011"))
#process.apvcyclephasemonitorEC02010 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhasesEC02010"))
#process.apvcyclephasemonitor2011 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhases2011"))
#process.apvcyclephasemonitor2010 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhases2010"))

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.wantEWHDepthHisto = cms.untracked.bool(True)
process.eventtimedistribution.wantDBXvsBX = cms.untracked.bool(True)
process.eventtimedistribution.wantOrbitvsBXincycle = cms.untracked.bool(True)
process.eventtimedistranydcs = process.eventtimedistribution.clone()
process.eventtimedistrdeco = process.eventtimedistribution.clone()
process.eventtimedistrpeak = process.eventtimedistribution.clone()
process.eventtimedistrIsoPairs5000 = process.eventtimedistribution.clone()
process.eventtimedistrIsoTriplets5000 = process.eventtimedistribution.clone()
process.eventtimedistrnonoisybins = process.eventtimedistribution.clone()
process.eventtimedistrnonoisybinsdeco = process.eventtimedistribution.clone()
process.eventtimedistrnonoisybinspeak = process.eventtimedistribution.clone()
#process.evttimedistrwithnophase = process.eventtimedistribution.clone(phasePartition = cms.untracked.string("None"))
#process.evttimedistrwithphaseEC02011 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02011"))
#process.evttimedistrwithphaseEC02010 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02010"))
#process.evttimedistrwithphase2011 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhases2011"))
#process.evttimedistrwithphase2010 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhases2010"))

# bad component analysis

import CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi

process.ssqoffbadstrips = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
process.ssqoffbadstrips.appendToDataLabel = cms.string("offlineBadStrips")
process.ssqoffbadstrips.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripBadFiberRcd'),tag=cms.string(''))
) 

process.ssqdcsbadmodules = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
process.ssqdcsbadmodules.appendToDataLabel = cms.string("dcsBadModules")
process.ssqdcsbadmodules.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetVOffRcd'),tag=cms.string(''))
) 


process.load("DPGAnalysis.SiStripTools.sistripqualityhistory_cfi")
process.ssqDCShistory = process.ssqhistory.clone()

process.ssqhistory.maxLSBeforeRebin = cms.untracked.uint32(10)
process.ssqhistory.startingLSFraction = cms.untracked.uint32(1)
process.ssqhistory.granularityMode=cms.untracked.uint32(3)
process.ssqhistory.monitoredSiStripQuality = cms.VPSet(
    cms.PSet( name = cms.string("OfflineBadStrips"), ssqLabel = cms.string("offlineBadStrips"))
    )# name= used in histos, ssqLabel= label of SiStripQuality object

process.ssqDCShistory.granularityMode=cms.untracked.uint32(0)
process.ssqDCShistory.monitoredSiStripQuality = cms.VPSet(
    cms.PSet( name = cms.string("DCSBadModules"), ssqLabel = cms.string("dcsBadModules"))
    )# name= used in histos, ssqLabel= label of SiStripQuality object



# cluster analysis

process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")

process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
process.ssclusmultinvestigator.scaleFactor = cms.untracked.int32(2)
process.ssclusmultinvestigator.numberOfBins = cms.untracked.int32(1000)
process.ssclusmultinvestdeco = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestpeak = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestIsoPairs5000 = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestIsoTriplets5000 = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestnonoisybins = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestnonoisybinsdeco = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestnonoisybinspeak = process.ssclusmultinvestigator.clone()


process.load("DPGAnalysis.SiStripTools.ssclusmulttimecorrelations_cfi")
process.ssclusmulttimecorrelations.historyProduct = cms.InputTag("consecutiveHEs")
process.ssclusmulttimecorrelations.scaleFactors = cms.untracked.vint32(2)
process.ssclusmulttimecorrelations.numberOfBins = cms.untracked.int32(1000)
process.ssclusmulttimecorrelations.runHisto = cms.untracked.bool(True)
process.ssclusmulttimecorrelations.dbx3Histo = cms.untracked.bool(True)
process.ssclusmulttimecorrelations.dbx3Histo3D = cms.untracked.bool(True)
process.ssclusmulttimecorrpeak = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrpeak.runHisto = cms.untracked.bool(False)
process.ssclusmulttimecorrdeco = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrdeco.runHisto = cms.untracked.bool(False)
process.ssclusmulttimecorrnonoisybins = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrnonoisybinspeak = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrnonoisybinspeak.runHisto = cms.untracked.bool(False)
process.ssclusmulttimecorrnonoisybinsdeco = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrnonoisybinsdeco.runHisto = cms.untracked.bool(False)
process.ssclusmulttimecorrIsoPairs5000 = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrIsoPairs5000.runHisto = cms.untracked.bool(False)
process.ssclusmulttimecorrIsoPairs5000.dbx3Histo = cms.untracked.bool(False)
process.ssclusmulttimecorrIsoPairs5000.dbx3Histo3D = cms.untracked.bool(False)
process.ssclusmulttimecorrIsoTriplets5000 = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrIsoTriplets5000.runHisto = cms.untracked.bool(False)
#process.ssclusmulttimecorrphaseEC02011 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02011"))
#process.ssclusmulttimecorrphaseEC02010 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02010"))
#process.ssclusmulttimecorrphase2011 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhases2011"))
#process.ssclusmulttimecorrphase2010 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhases2010"))
#process.ssclusmulttimecorrnophase = process.ssclusmulttimecorrelations.clone()
#process.ssclusmulttimecorrnophase.wantedSubDets = cms.untracked.VPSet(    
# cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64), phasePartition = cms.untracked.string("None")),
# cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/64), phasePartition = cms.untracked.string("None")),
# cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID"), binMax = cms.int32( 565248/64), phasePartition = cms.untracked.string("None")),
# cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB"), binMax = cms.int32(3303936/64), phasePartition = cms.untracked.string("None")),
# cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"), binMax = cms.int32(3866624/64), phasePartition = cms.untracked.string("None"))
# )#

#

process.load("DPGAnalysis.Skims.DetStatus_cfi")
process.dcsstatus.DetectorType = cms.vstring("TIBTID","TOB","TECp","TECm") 


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('apvcyclephaseTest.root')
                                   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
#-------------------------------------------------------------------------
#process.poolDBESSource = cms.ESSource("PoolDBESSource",
#   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#   DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(1),
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.untracked.string('runnumber'),
#    connect = cms.string('sqlite_file:apvphaseoffsets_forHLT.db'),
#    appendToDataLabel = cms.string("apvphaseoffsets"),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiStripConfObjectRcd'),
#        tag = cms.string('SiStripAPVPhaseOffsets_real_v1')
#    ))
#)
#process.es_prefer = cms.ESPrefer("PoolDBESSource","poolDBESSource")
#-------------------------------------------------------------------------


#

process.panydcs = cms.Path(process.consecutiveHEs
                           + process.APVPhases
                           + process.l1TSDebugger
                           + process.ssqDCShistory
                           + process.eventtimedistranydcs
                           + process.l1abcdebuanydcs
                           )

process.p0 = cms.Path(process.consecutiveHEs
                      + process.APVPhases
                      + process.l1TSDebugger
#                      + process.APVPhasesEC02011 + process.APVPhasesEC02010 + process.APVPhases2011 + process.APVPhases2010
                      + process.dcsstatus
                      + process.apvcyclephasemonitor
                      + process.ssqhistory
#                      + process.apvcyclephasemonitorEC02011 + process.apvcyclephasemonitorEC02010
#                      + process.apvcyclephasemonitor2011 + process.apvcyclephasemonitor2010
                      + process.eventtimedistribution
                      + process.l1abcdebugger
#                      + process.evttimedistrwithnophase
#                      + process.evttimedistrwithphaseEC02011 + process.evttimedistrwithphaseEC02011
#                      + process.evttimedistrwithphase2011 + process.evttimedistrwithphase2010
                      + process.ssclustermultprod + process.ssclusmultinvestigator
                      + process.ssclusmulttimecorrelations
#                      + process.ssclusmulttimecorrnophase
#                      + process.ssclusmulttimecorrphaseEC02011 + process.ssclusmulttimecorrphaseEC02010
#                      + process.ssclusmulttimecorrphase2011 + process.ssclusmulttimecorrphase2010)
                      )

process.ppeak = cms.Path(process.consecutiveHEs
                         + process.APVPhases
                         + process.l1TSDebugger
                         + process.dcsstatus
                         + process.PeakEvents
                         + process.eventtimedistrpeak
                         + process.ssclustermultprod + process.ssclusmultinvestpeak
                         + process.ssclusmulttimecorrpeak
                         )

process.pdeco = cms.Path(process.consecutiveHEs
                         + process.APVPhases
                         + process.l1TSDebugger
                         + process.dcsstatus
                         + process.DecoEvents
                         + process.eventtimedistrdeco
                         + process.ssclustermultprod + process.ssclusmultinvestdeco
                         + process.ssclusmulttimecorrdeco
                         )

process.pIsoPairs5000 = cms.Path(process.consecutiveHEs
                                 + process.APVPhases
                                 + process.l1TSDebugger
                                 + process.dcsstatus
                                 + process.isoPairs5000
                                 + process.ssclustermultprod 
                                 + process.eventtimedistrIsoPairs5000
                                 + process.ssclusmultinvestIsoPairs5000
                                 + process.ssclusmulttimecorrIsoPairs5000
                                 )

process.pIsoTriplets5000 = cms.Path(process.consecutiveHEs
                                    + process.APVPhases
                                    + process.l1TSDebugger
                                    + process.dcsstatus
                                    + process.isoTriplets5000
                                    + process.ssclustermultprod 
                                    + process.eventtimedistrIsoTriplets5000
                                    + process.ssclusmultinvestIsoTriplets5000
                                    + process.ssclusmulttimecorrIsoTriplets5000
                                    )

process.pnonoisybins = cms.Path(process.consecutiveHEs
                                + process.APVPhases
                                + process.l1TSDebugger
                                + process.dcsstatus
                                + process.seqNoNoisyBins
                                + process.ssclustermultprod 
                                + process.eventtimedistrnonoisybins
                                + process.ssclusmultinvestnonoisybins
                                + process.ssclusmulttimecorrnonoisybins
                                )

process.pnonoisybinspeak = cms.Path(process.consecutiveHEs
                                + process.APVPhases
                                + process.l1TSDebugger
                                + process.dcsstatus
                                + process.PeakEvents    
                                + process.seqNoNoisyBins
                                + process.ssclustermultprod 
                                + process.eventtimedistrnonoisybinspeak
                                + process.ssclusmultinvestnonoisybinspeak
                                + process.ssclusmulttimecorrnonoisybinspeak
                                )

process.pnonoisybinsdeco = cms.Path(process.consecutiveHEs
                                + process.APVPhases
                                + process.l1TSDebugger
                                + process.dcsstatus
                                + process.DecoEvents    
                                + process.seqNoNoisyBins
                                + process.ssclustermultprod 
                                + process.eventtimedistrnonoisybinsdeco
                                + process.ssclusmultinvestnonoisybinsdeco
                                + process.ssclusmulttimecorrnonoisybinsdeco
                                )

process.schedule = cms.Schedule(process.panydcs,process.p0)

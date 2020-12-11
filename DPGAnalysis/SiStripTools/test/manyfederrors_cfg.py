import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("ManyFEDErrors")

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
process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
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

process.MessageLogger.debugModules=cms.untracked.vstring("eventtimedistribution")
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
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
process.load("DPGAnalysis.SiStripTools.l1TSDebugger_cfi")

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")
process.eventtimedistribution.dbxHistosParams = cms.untracked.VPSet(
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(0),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(1),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(2),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(3),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(4),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(5),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(6),secondEvent=cms.uint32(7)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(0),secondEvent=cms.uint32(6)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(1),secondEvent=cms.uint32(6)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(2),secondEvent=cms.uint32(6)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(3),secondEvent=cms.uint32(6)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(4),secondEvent=cms.uint32(6)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(5),secondEvent=cms.uint32(6)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(0),secondEvent=cms.uint32(5)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(1),secondEvent=cms.uint32(5)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(2),secondEvent=cms.uint32(5)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(3),secondEvent=cms.uint32(5)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(4),secondEvent=cms.uint32(5)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(0),secondEvent=cms.uint32(4)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(1),secondEvent=cms.uint32(4)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(2),secondEvent=cms.uint32(4)),
    cms.PSet(nbins=cms.int32(2000),min=cms.double(-0.5),max=cms.double(3999.5),firstEvent=cms.uint32(3),secondEvent=cms.uint32(4)),
    cms.PSet(nbins=cms.int32(1000),min=cms.double(-0.5),max=cms.double(999.5),firstEvent=cms.uint32(0),secondEvent=cms.uint32(1)),
    )

process.eventtimedistrmanyfederrorsallthr = process.eventtimedistribution.clone()
process.eventtimedistrmanyfederrorshighthr = process.eventtimedistribution.clone()
process.eventtimedistrmanyfederrorsmidthr = process.eventtimedistribution.clone()
process.eventtimedistrmanyfederrorslowthr = process.eventtimedistribution.clone()
process.eventtimedistrnomanyfederrors = process.eventtimedistribution.clone()

process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases + process.l1TSDebugger)

process.apvcyclephasemonitor = cms.EDAnalyzer('APVCyclePhaseMonitor',
                                              apvCyclePhaseCollection = cms.InputTag("APVPhases"),
                                              maxLSBeforeRebin = cms.untracked.uint32(250),
                                              startingLSFraction = cms.untracked.uint32(16),
                                              selectedPartitions = cms.untracked.vstring("TI","TO","TP","TM"),
                                              selectedVectorPartitions = cms.untracked.vstring("Any")
)	


process.seqEventHistory = cms.Sequence(process.eventtimedistribution + process.apvcyclephasemonitor)
process.seqEventHistoryManyFEDErrorsAllThr = cms.Sequence(process.eventtimedistrmanyfederrorsallthr)
process.seqEventHistoryManyFEDErrorsHighThr = cms.Sequence(process.eventtimedistrmanyfederrorshighthr)
process.seqEventHistoryManyFEDErrorsMidThr = cms.Sequence(process.eventtimedistrmanyfederrorsmidthr)
process.seqEventHistoryManyFEDErrorsLowThr = cms.Sequence(process.eventtimedistrmanyfederrorslowthr)
process.seqEventHistoryNoManyFEDErrors = cms.Sequence(process.eventtimedistrnomanyfederrors)

process.seqProducers = cms.Sequence(process.seqEventHistoryReco)

#do not ignore APVe mismatches

process.siStripDigis.DoAPVEmulatorCheck = cms.bool(True)
#process.siStripDigis.ErrorThreshold = cms.uint32(30000)

#process.siStripDigisMidThr = process.siStripDigis.clone(ErrorThreshold = cms.uint32(7174))
#process.siStripDigisLowThr = process.siStripDigis.clone(ErrorThreshold = cms.uint32(2800))
#process.siStripDigisAllThr = process.siStripDigis.clone(ErrorThreshold = cms.uint32(35500))

#process.seqSiStripDigis = cms.Sequence(process.siStripDigis + process.siStripDigisLowThr + process.siStripDigisMidThr + process.siStripDigisAllThr)
process.seqSiStripDigis = cms.Sequence(process.siStripDigis)

process.seqRECO = cms.Sequence(process.seqSiStripDigis +
                               process.scalersRawToDigi +
                               process.logErrorHarvester 
                               )

#process.logErrorTooManyErrors = cms.EDFilter("LogErrorEventFilter",
#                                     src = cms.InputTag("logErrorHarvester"),
#                                     maxErrorFractionInLumi = cms.double(1.0),
#                                     maxErrorFractionInRun  = cms.double(1.0),
#                                     maxSavedEventsPerLumiAndError = cms.uint32(1000000),
#                                     #                                        categoriesToIgnore = cms.vstring("HLTConfigProvider","FastCloningDisabled")
#                                     categoriesToWatch = cms.vstring("TooManyErrors"),
#                                     modulesToWatch = cms.vstring("SiStripRawToDigiModule:siStripDigis")
#                                     
#                                     )

#process.logErrorTooManyErrorsLowThr = process.logErrorTooManyErrors.clone(modulesToWatch = cms.vstring("SiStripRawToDigiModule:siStripDigisLowThr"))
#process.logErrorTooManyErrorsMidThr = process.logErrorTooManyErrors.clone(modulesToWatch = cms.vstring("SiStripRawToDigiModule:siStripDigisMidThr"))
#process.logErrorTooManyErrorsAllThr = process.logErrorTooManyErrors.clone(modulesToWatch = cms.vstring("SiStripRawToDigiModule:siStripDigisAllThr"))

process.filterFEDBadModuleHighThr = cms.EDFilter("FEDBadModuleFilter",
                                                 collectionName=cms.InputTag("siStripDigis"),
                                                 badModThr = cms.uint32(30000),
                                                 wantedHisto=cms.untracked.bool(False),
                                                 maxLSBeforeRebin = cms.untracked.uint32(250),
                                                 startingLSFraction = cms.untracked.uint32(16)
                                                 )
process.filterFEDBadModuleAllThr = process.filterFEDBadModuleHighThr.clone(badModThr = cms.uint32(35500),wantedHisto=cms.untracked.bool(True))
process.filterFEDBadModuleMidThr = process.filterFEDBadModuleHighThr.clone(badModThr = cms.uint32(7174),wantedHisto=cms.untracked.bool(False))
process.filterFEDBadModuleLowThr = process.filterFEDBadModuleHighThr.clone(badModThr = cms.uint32(2000),wantedHisto=cms.untracked.bool(False))

process.badModuleFED74 = cms.EDFilter("FEDBadModuleFilter",
                                      collectionName=cms.InputTag("siStripDigis"),
                                      badModThr = cms.uint32(0),
                                      wantedHisto=cms.untracked.bool(True),
                                      moduleList = cms.untracked.vuint32(369158204),
                                      maxLSBeforeRebin = cms.untracked.uint32(250),
                                      startingLSFraction = cms.untracked.uint32(16)
                                      )
process.badModuleFED102 = cms.EDFilter("FEDBadModuleFilter",
                                       collectionName=cms.InputTag("siStripDigis"),
                                       badModThr = cms.uint32(0),
                                       wantedHisto=cms.untracked.bool(True),
                                       moduleList = cms.untracked.vuint32(369141837,369141833,369141829),
                                       maxLSBeforeRebin = cms.untracked.uint32(250),
                                       startingLSFraction = cms.untracked.uint32(16)
                                      )
process.badModuleFED108 = cms.EDFilter("FEDBadModuleFilter",
                                       collectionName=cms.InputTag("siStripDigis"),
                                       badModThr = cms.uint32(0),
                                       wantedHisto=cms.untracked.bool(True),
                                       moduleList = cms.untracked.vuint32(369142074,369142077,369142078),
                                       maxLSBeforeRebin = cms.untracked.uint32(250),
                                       startingLSFraction = cms.untracked.uint32(16)
                                       )


#process.seqTooManyErrorsAllThr = cms.Sequence(process.logErrorTooManyErrorsAllThr)
#process.seqTooManyErrorsHighThr = cms.Sequence(~process.logErrorTooManyErrorsAllThr + process.logErrorTooManyErrors)
#process.seqTooManyErrorsMidThr = cms.Sequence(~process.logErrorTooManyErrorsAllThr + ~process.logErrorTooManyErrors + process.logErrorTooManyErrorsMidThr)
#process.seqTooManyErrorsLowThr = cms.Sequence(~process.logErrorTooManyErrorsAllThr + ~process.logErrorTooManyErrors + ~process.logErrorTooManyErrorsMidThr +
#                                              process.logErrorTooManyErrorsLowThr)
#process.seqNoTooManyErrors = cms.Sequence(~process.logErrorTooManyErrorsAllThr + ~process.logErrorTooManyErrors + ~process.logErrorTooManyErrorsMidThr +
#                                          ~process.logErrorTooManyErrorsLowThr)

process.seqTooManyErrorsAllThr = cms.Sequence(process.filterFEDBadModuleAllThr)
process.seqTooManyErrorsHighThr = cms.Sequence(~process.filterFEDBadModuleAllThr + process.filterFEDBadModuleHighThr)
process.seqTooManyErrorsMidThr = cms.Sequence(~process.filterFEDBadModuleAllThr + ~process.filterFEDBadModuleHighThr +
                                              process.filterFEDBadModuleMidThr)
process.seqTooManyErrorsLowThr = cms.Sequence(~process.filterFEDBadModuleAllThr + ~process.filterFEDBadModuleHighThr +
                                              ~process.filterFEDBadModuleMidThr +process.filterFEDBadModuleLowThr)
process.seqNoTooManyErrors = cms.Sequence(~process.filterFEDBadModuleAllThr + ~process.filterFEDBadModuleHighThr +
                                          ~process.filterFEDBadModuleMidThr + ~process.filterFEDBadModuleLowThr)


process.load("DQM.SiStripCommon.TkHistoMap_cff")

process.load("DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff")

process.siStripFEDMonitor.BadMajorityInPartitionHistogramConfig.Enabled = True

process.siStripFEDMonitorAllThr = process.siStripFEDMonitor.clone(  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedSummaryAllThr'))
process.siStripFEDMonitorHighThr = process.siStripFEDMonitor.clone(  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedSummaryHighThr'))
process.siStripFEDMonitorMidThr = process.siStripFEDMonitor.clone(  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedSummaryMidThr'))
process.siStripFEDMonitorLowThr = process.siStripFEDMonitor.clone(  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedSummaryLowThr'))
process.siStripFEDMonitorNoErr = process.siStripFEDMonitor.clone(  HistogramFolderName = cms.untracked.string('SiStrip/ReadoutView/FedSummaryNoErr'))

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.EventContent.EventContent_cff")

process.DQMoutput = cms.OutputModule("PoolOutputModule",
                                     splitLevel = cms.untracked.int32(0),
                                     outputCommands = process.DQMEventContent.outputCommands,
                                     fileName = cms.untracked.string('manyfederrors_DQM.root'),
                                     dataset = cms.untracked.PSet(
            filterName = cms.untracked.string(''),
                    dataTier = cms.untracked.string('DQM')
                )
                                                  )

process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
             

process.p0 = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.siStripFEDMonitor +
    process.seqEventHistory +
    process.badModuleFED74 +
    process.badModuleFED102 +
    process.badModuleFED108 
    )

process.pnomanyfederrors = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.seqNoTooManyErrors +
    process.siStripFEDMonitorNoErr +
    process.seqEventHistoryNoManyFEDErrors 
    )

process.pmanyfederrorslowthr = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.seqTooManyErrorsLowThr +
    process.siStripFEDMonitorLowThr +
    process.seqEventHistoryManyFEDErrorsLowThr 
    )

process.pmanyfederrorsmidthr = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.seqTooManyErrorsMidThr +
    process.siStripFEDMonitorMidThr +
    process.seqEventHistoryManyFEDErrorsMidThr 
    )

process.pmanyfederrorshighthr = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.seqTooManyErrorsHighThr +
    process.siStripFEDMonitorHighThr +
    process.seqEventHistoryManyFEDErrorsHighThr 
    )

process.pmanyfederrorsallthr = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.seqTooManyErrorsAllThr +
    process.siStripFEDMonitorAllThr +
    process.seqEventHistoryManyFEDErrorsAllThr 
    )


#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('ManyFEDErrors.root')
                                   )

#print process.dumpPython()

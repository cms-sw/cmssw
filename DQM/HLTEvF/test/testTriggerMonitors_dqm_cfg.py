import FWCore.ParameterSet.Config as cms

# VarParsing
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('nThreads', 4, options.multiplicity.singleton, options.varType.int, 'number of threads')
options.register('nStreams', 0, options.multiplicity.singleton, options.varType.int, 'number of streams')
options.register('globalTag', 'auto:run3_hlt_relval', options.multiplicity.singleton, options.varType.string, 'name of GlobalTag')
options.setDefault('inputFiles', [
  '/store/data/Run2022B/HLTPhysics/RAW/v1/000/355/456/00000/69b26b27-4bd1-4524-bc18-45f7b9b5e076.root',
])
options.setDefault('maxEvents', 200)
options.setType('outputFile', options.varType.string)
options.setDefault('outputFile', 'DQMIO.root')
options.parseArguments()

# Process
process = cms.Process('DQM')

process.options.numberOfThreads = options.nThreads
process.options.numberOfStreams = options.nStreams
process.maxEvents.input = options.maxEvents

# Source (EDM input)
process.source = cms.Source('PoolSource',
  fileNames = cms.untracked.vstring(options.inputFiles),
  inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_rawDataCollector__*',
    'keep edmTriggerResults_TriggerResults__HLT',
  )
)

# DQMStore (Service)
process.load('DQMServices.Core.DQMStore_cfi')

# MessageLogger (Service)
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# FastTimerService (Service)
from HLTrigger.Timer.FastTimerService_cfi import FastTimerService as _FastTimerService
process.FastTimerService = _FastTimerService.clone(
  dqmTimeRange = 2000,
  enableDQM = True,
  enableDQMTransitions = True,
  enableDQMbyLumiSection = True,
  enableDQMbyModule = True,
  enableDQMbyPath = True,
  enableDQMbyProcesses = True
)
process.MessageLogger.FastReport = dict()

# ThroughputService (Service)
from HLTrigger.Timer.ThroughputService_cfi import ThroughputService as _ThroughputService
process.ThroughputService = _ThroughputService.clone(
  dqmPathByProcesses = True,
  timeRange = 60000,
  timeResolution = 5.828
)
process.MessageLogger.ThroughputService = dict()

# GlobalTag (ESSource)
from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(globaltag = options.globalTag)

# EventData modules
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis as _gtStage2Digis
process.gtStage2Digis = _gtStage2Digis.clone()

from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import scalersRawToDigi as _scalersRawToDigi
process.scalersRawToDigi = _scalersRawToDigi.clone()

from EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi import onlineMetaDataRawToDigi as _onlineMetaDataDigis
process.onlineMetaDataDigis = _onlineMetaDataDigis.clone()

from DQM.HLTEvF.triggerRatesMonitor_cfi import triggerRatesMonitor as _triggerRatesMonitor
process.triggerRatesMonitor = _triggerRatesMonitor.clone(
  hltResults = 'TriggerResults::HLT'
)

from DQM.HLTEvF.triggerBxMonitor_cfi import triggerBxMonitor as _triggerBxMonitor
process.triggerBxMonitor = _triggerBxMonitor.clone(
  hltResults = 'TriggerResults::HLT'
)

from DQM.HLTEvF.triggerBxVsOrbitMonitor_cfi import triggerBxVsOrbitMonitor as _triggerBxVsOrbitMonitor
process.triggerBxVsOrbitMonitor = _triggerBxVsOrbitMonitor.clone(
  hltResults = 'TriggerResults::HLT'
)

from DQM.HLTEvF.lumiMonitor_cfi import lumiMonitor as _lumiMonitor
process.lumiMonitor = _lumiMonitor.clone(
  scalers = 'scalersRawToDigi',
  onlineMetaDataDigis = 'onlineMetaDataDigis'
)

from DQM.HLTEvF.psMonitoring_cfi import psMonitoring as _psColumnMonitor
process.psColumnMonitor = _psColumnMonitor.clone(
  ugtBXInputTag = 'gtStage2Digis',
  histoPSet = dict(
    psColumnPSet = dict(
      nbins = 20
    )
  )
)

# Output module (file in DQM format)
process.dqmOutput = cms.OutputModule('DQMRootOutputModule',
  fileName = cms.untracked.string(options.outputFile)
)

# EndPath
process.endp = cms.EndPath(
    process.gtStage2Digis
  + process.scalersRawToDigi
  + process.onlineMetaDataDigis
  + process.triggerRatesMonitor
  + process.triggerBxMonitor
  + process.triggerBxVsOrbitMonitor
  + process.lumiMonitor
  + process.psColumnMonitor
  + process.dqmOutput
)

import FWCore.ParameterSet.Config as cms

# VarParsing
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('nThreads', 4, options.multiplicity.singleton, options.varType.int, 'number of threads')
options.register('nStreams', 0, options.multiplicity.singleton, options.varType.int, 'number of streams')
options.setDefault('inputFiles', ['file:DQMIO.root'])
options.parseArguments()

# Process
process = cms.Process('HARVESTING')

process.options.numberOfThreads = options.nThreads
process.options.numberOfStreams = options.nStreams
process.options.numberOfConcurrentLuminosityBlocks = 1

# Source (DQM input)
process.source = cms.Source('DQMRootSource',
  fileNames = cms.untracked.vstring(options.inputFiles)
)

# DQMStore (Service)
process.load('DQMServices.Core.DQMStore_cfi')

# MessageLogger (Service)
process.load('FWCore.MessageLogger.MessageLogger_cfi')

# Harvesting modules

# FastTimerService client
from HLTrigger.Timer.fastTimerServiceClient_cfi import fastTimerServiceClient as _fastTimerServiceClient
process.fastTimerServiceClient = _fastTimerServiceClient.clone(
  dqmPath = 'HLT/TimerService',
  # timing VS lumi
  doPlotsVsOnlineLumi = True,
  doPlotsVsPixelLumi = False,
  onlineLumiME = dict(
    folder = 'HLT/LumiMonitoring',
    name   = 'lumiVsLS',
    nbins  = 5000,
    xmin   = 0,
    xmax   = 20000
  )
)

# ThroughputService client
from HLTrigger.Timer.throughputServiceClient_cfi import throughputServiceClient as _throughputServiceClient
process.throughputServiceClient = _throughputServiceClient.clone(
  dqmPath = 'HLT/Throughput'
)

# PS column VS lumi
from DQM.HLTEvF.dqmCorrelationClient_cfi import dqmCorrelationClient as _dqmCorrelationClient
process.psColumnVsLumi = _dqmCorrelationClient.clone(
   me = dict(
      folder = 'HLT/PSMonitoring',
      name = 'psColumnVSlumi',
      doXaxis = True,
      nbinsX = 5000,
      xminX = 0,
      xmaxX = 20000,
      doYaxis = False,
      nbinsY = 8,
      xminY = 0,
      xmaxY = 8
   ),
   me1 = dict(
      folder = 'HLT/LumiMonitoring',
      name = 'lumiVsLS',
      profileX = True
   ),
   me2 = dict(
      folder = 'HLT/PSMonitoring',
      name = 'psColumnIndexVsLS',
      profileX = True
   )
)

from DQM.HLTEvF.triggerRatesMonitorClient_cfi import triggerRatesMonitorClient as _triggerRatesMonitorClient
process.triggerRatesMonitorClient = _triggerRatesMonitorClient.clone(
  dqmPath = 'HLT/TriggerRates'
)

# Output module (file in ROOT format)
from DQMServices.Components.DQMFileSaver_cfi import dqmSaver as _dqmSaver
process.dqmSaver = _dqmSaver.clone(
  workflow = '/HLTEvF/TestTriggerMonitors/'+process.name_()
)

# EndPath
process.endp = cms.EndPath(
    process.fastTimerServiceClient
  + process.throughputServiceClient
  + process.psColumnVsLumi
  + process.triggerRatesMonitorClient
  + process.dqmSaver
)

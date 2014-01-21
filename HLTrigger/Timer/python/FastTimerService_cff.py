import FWCore.ParameterSet.Config as cms

# SCAL (for online luminosity)
scalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
  scalersInputTag = cms.InputTag( "rawDataCollector" )
)

# LumiProducer and LumiCorrectionSource (for offline luminosity)
DBService = cms.Service('DBService')

lumiProducer = cms.EDProducer('LumiProducer',
  connect       = cms.string('frontier://LumiProd/CMS_LUMI_PROD'),
  lumiversion   = cms.untracked.string(''),
  ncacheEntries = cms.untracked.uint32(5),
)

LumiCorrectionSource = cms.ESSource( "LumiCorrectionSource",
  connect       = cms.string('frontier://LumiCalc/CMS_LUMI_PROD'),
)

# Pixel clusters
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
siPixelDigis.InputLabel = cms.InputTag('rawDataCollector')

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters

# FastTimerService
from HLTrigger.Timer.FastTimerService_cfi import *
from HLTrigger.Timer.fastTimerServiceClient_cfi import *

from HLTrigger.Timer.ftsLuminosityFromScalers_cfi import *
ftsLuminosityFromScalers.name  = 'onlineluminosity'
ftsLuminosityFromScalers.title = 'online instantaneous luminosity'
ftsLuminosityFromScalers.range = 1.6e34

from HLTrigger.Timer.ftsPileupFromScalers_cfi import *
ftsPileupFromScalers.name  = 'onlinepileup'
ftsPileupFromScalers.title = 'online pileup'
ftsPileupFromScalers.range = 80

from HLTrigger.Timer.ftsLuminosityFromLumiSummary_cfi import *
ftsLuminosityFromLumiSummary.name  = 'offlineluminosity'
ftsLuminosityFromLumiSummary.title = 'offline instantaneous luminosity'
ftsLuminosityFromLumiSummary.range = 1.6e34

from HLTrigger.Timer.ftsPileupFromLumiSummary_cfi import *
ftsPileupFromLumiSummary.name  = cms.string('offlinepileup')
ftsPileupFromLumiSummary.title = cms.string('offline pileup')
ftsPileupFromLumiSummary.range = 80

from HLTrigger.Timer.ftsLuminosityFromPixelClusters_cfi import *

# DQM file saver
dqmFileSaver = cms.EDAnalyzer( "DQMFileSaver",
#   producer          = cms.untracked.string('DQM'),
#   version           = cms.untracked.int32(1),
#   referenceRequireStatus = cms.untracked.int32(100),
#   runIsComplete     = cms.untracked.bool(False),
#   referenceHandling = cms.untracked.string('all'),
    convention        = cms.untracked.string( "Offline" ),
    workflow          = cms.untracked.string( "/HLT/FastTimerService/All" ),
    dirName           = cms.untracked.string( "." ),
    saveByRun         = cms.untracked.int32(1),
    saveByLumiSection = cms.untracked.int32(-1),
    saveByEvent       = cms.untracked.int32(-1),
    saveByTime        = cms.untracked.int32(-1),
    saveByMinute      = cms.untracked.int32(-1),
    saveAtJobEnd      = cms.untracked.bool(False),
    forceRunNumber    = cms.untracked.int32(-1),
)

DQMFileSaverOutput = cms.EndPath(
    scalersRawToDigi + ftsLuminosityFromScalers + ftsPileupFromScalers +
    lumiProducer + ftsLuminosityFromLumiSummary + ftsPileupFromLumiSummary +
    siPixelDigis + siPixelClusters + ftsLuminosityFromPixelClusters +
    fastTimerServiceClient + dqmFileSaver )

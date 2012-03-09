# The following comments couldn't be translated into the new config version:

#   source = EventStreamHttpReader
#          {
#             string sourceURL = "http://cmsdisk1.cms:48500/urn:xdaq-application:service=storagemanager"
#             int32 max_event_size = 7000000
#             int32 max_queue_depth = 5
#             untracked string consumerName = "Test Consumer"
#             untracked string consumerPriority = "normal"
#             untracked int32 headerRetryInterval = 3  // seconds
#             untracked double maxEventRequestRate = 100.  // hertz
#untracked PSet SelectEvents = { vstring SelectEvents= {"p2"} }
#             untracked PSet SelectEvents = { vstring SelectEvents= {"*"} }
#          }
#	source = PoolSource{
#	        untracked vstring fileNames={
#        "file:/nfshome0/berryhil/CMSSW_1_6_0_DAQ1/src/GREJDigi.root"
#		"file:testSourceCardTextToRctDigi.root"
#"file:/nfshome0/berryhil/CMSSW_1_4_0_DAQ1/src/testGt_Unpacker_output.root"
#		}
#               untracked bool debugVebosity = false
#                untracked uint32 debugVerbosity = 1 
#		untracked int32 maxEvents = -1
#        }
#
#  DQM SERVICES
#

# Message Logger service
#include "FWCore/MessageService/data/MessageLogger.cfi"
# uncomment / comment messages with DEBUG mode to run in DEBUG mode

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
#
#  DQM SOURCES
#
process.load("DQM.L1TMonitor.L1TMonitor_cff")

process.load("DQM.L1TMonitor.debug_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("EventStreamHttpReader",
    sourceURL = cms.string('http://cmsroc9.fnal.gov:50082/urn:xdaq-application:lid=29'), ##cmsroc9.fnal.gov:50082/urn:xdaq-application:lid=29"

    consumerPriority = cms.untracked.string('normal'),
    #   string sourceURL = "http://lhc01n02.fnal.gov:50082/urn:xdaq-application:lid=29"
    #   string sourceURL = "http://lhc01n02.fnal.gov:50082/urn:xdaq-application:lid=29"
    #  string sourceURL = "http://lhc01n02.fnal.gov:50082/urn:xdaq-application:lid=29"
    max_event_size = cms.int32(7000000),
    consumerName = cms.untracked.string('Test Consumer'),
    max_queue_depth = cms.int32(5),
    maxEventRequestRate = cms.untracked.double(2.5), ## hertz

    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*')
    ),
    headerRetryInterval = cms.untracked.int32(3) ## seconds

)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.DQMShipMonitoring = cms.Service("DQMShipMonitoring",
    # event-period for shipping monitoring to collector (default: 25)
    period = cms.untracked.uint32(5)
)

process.MonitorDaemon = cms.Service("MonitorDaemon",
    # at cmsuaf
    #       untracked string DestinationAddress = "cmsroc2"
    # on cms online cluster
    #       untracked string DestinationAddress = "rubus2d16-13"
    # on lxplus
    DestinationAddress = cms.untracked.string('lxplus211.cern.ch'),
    SendPort = cms.untracked.int32(9090),
    #       untracked   bool AutoInstantiate    = true
    reconnect_delay = cms.untracked.int32(5),
    NameAsSource = cms.untracked.string('GlobalDQM'),
    UpdateDelay = cms.untracked.int32(1000)
)

process.MessageLogger = cms.Service("MessageLogger",
    testGt_Unpacker = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'), ## DEBUG mode 

        DEBUG = cms.untracked.PSet( ## DEBUG mode, all messages  

            limit = cms.untracked.int32(-1)
        ),
        #        untracked PSet DEBUG = { untracked int32 limit = 10}  // DEBUG mode, max 10 messages  
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('testGt_Unpacker')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)


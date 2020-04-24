import FWCore.ParameterSet.Config as cms

# DQMStore service
from DQMServices.Core.DQMStore_cfi import DQMStore

# FastTimerService client
from HLTrigger.Timer.fastTimerServiceClient_cfi import fastTimerServiceClient
fastTimerServiceClient.dqmPath  = "HLT/TimerService"

# ThroughputService client
from HLTrigger.Timer.throughputServiceClient_cfi import throughputServiceClient
throughputServiceClient.dqmPath = "HLT/Throughput"

# run the harveting modules and the DQMFileSaver
HLTHarvestingSequence = cms.Sequence( fastTimerServiceClient + throughputServiceClient )

import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineHeavyIons_cff import *

#	remove TP Task and Raw Task from MC workflow for hcal
#	identical to pp MC Workflow
hcalOfflineSourceSequence.remove(tpTask)
hcalOfflineSourceSequence.remove(rawTask)

siStripFEDCheck.RawDataTag = 'rawDataCollector'
siStripFEDMonitor.RawDataTag = 'rawDataCollector'
SiPixelHLTSource.RawInput = 'rawDataCollector'
dqmCSCClient.InputObjects = 'rawDataCollector'
cscMonitor.FEDRawDataCollectionTag = 'rawDataCollector'
dtDataIntegrityUnpacker.inputLabel = 'rawDataCollector'
#l1tfed.rawTag = 'rawDataCollector'  # not needed until trigger DQM is enabled
ecalMonitorTask.collectionTags.TrigPrimEmulDigi = 'simEcalTriggerPrimitiveDigis'
ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = 'rawDataCollector'
ecalPreshowerRawDataTask.FEDRawDataCollection = 'rawDataCollector'

# L1 Trigger - remove emulator and adapt labels for private unpacking
from DQMOffline.L1Trigger.L1TriggerDqmOfflineMC_cff import *

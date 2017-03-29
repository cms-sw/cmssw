import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_cff import *

#	remove Trigger Primtive Task from hcal's sequence
hcalOfflineSourceSequence.remove(tpTask)
hcalOfflineSourceSequence.remove(rawTask)

siStripFEDCheck.RawDataTag = 'rawDataCollector'
siStripFEDMonitor.RawDataTag = 'rawDataCollector'
SiPixelHLTSource.RawInput = 'rawDataCollector'
dqmCSCClient.InputObjects = 'rawDataCollector'
dtDataIntegrityUnpacker.inputLabel = 'rawDataCollector'
#l1tfed.rawTag = 'rawDataCollector'
ecalMonitorTask.collectionTags.TrigPrimEmulDigi = 'simEcalTriggerPrimitiveDigis'
ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = 'rawDataCollector'
ecalPreshowerRawDataTask.FEDRawDataCollection = 'rawDataCollector'
castorOfflineMonitor.rawLabel = 'rawDataCollector'
cscMonitor.FEDRawDataCollectionTag = 'rawDataCollector'

# L1 Trigger - remove emulator and adapt labels for private unpacking
from DQMOffline.L1Trigger.L1TriggerDqmOfflineMC_cff import *

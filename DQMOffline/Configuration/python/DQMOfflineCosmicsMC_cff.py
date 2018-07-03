import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineCosmics_cff import *

#	remove RAW Task from MC workflow
hcalOfflineSourceSequence.remove(rawTask)

siStripFEDCheck.RawDataTag = 'rawDataCollector'
siStripFEDMonitor.RawDataTag = 'rawDataCollector'
SiPixelHLTSource.RawInput = 'rawDataCollector'
dqmCSCClient.InputObjects = 'rawDataCollector'
cscMonitor.FEDRawDataCollectionTag = 'rawDataCollector'
dtDataIntegrityUnpacker.inputLabel = 'rawDataCollector'
#l1tfed.rawTag = 'rawDataCollector'
ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = 'rawDataCollector'
ecalPreshowerRawDataTask.FEDRawDataCollection = 'rawDataCollector'


import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineCosmics_cff import *

siStripFEDCheck.RawDataTag = 'rawDataCollector'
SiPixelHLTSource.RawInput = 'rawDataCollector'
dqmCSCClient.InputObjects = 'rawDataCollector'
cscDQMEvF.InputObjects = 'rawDataCollector'
ecalBarrelHltTask.FEDRawDataCollection = 'rawDataCollector'
ecalBarrelSelectiveReadoutTask.FEDRawDataCollection = 'rawDataCollector'
ecalEndcapHltTask.FEDRawDataCollection = 'rawDataCollector'
ecalEndcapSelectiveReadoutTask.FEDRawDataCollection = 'rawDataCollector'
dtDataIntegrityUnpacker.inputLabel = cms.untracked.InputTag('rawDataCollector') 

# this is a TEMPORARY HUGLY hack until the DT map gets fixed
DQMOfflineCosmics.remove(dtDataIntegrityUnpacker)


import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_cff import *

siStripFEDCheck.RawDataTag = 'rawDataCollector'
dqmClient.InputObjects = 'rawDataCollector'
#dqmCSCClient.InputObjects = 'rawDataCollector'
cscDQMEvF.InputObjects = 'rawDataCollector'
ecalBarrelHltTask.FEDRawDataCollection = 'rawDataCollector'
ecalEndcapHltTask.FEDRawDataCollection = 'rawDataCollector'
dtDataIntegrityUnpacker.inputLabel = cms.untracked.InputTag('rawDataCollector')

# this is a TEMPORARY HUGLY hack until the L1TGMT gets fixed
DQMOffline.remove(dtDataIntegrityUnpacker)


import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_cff import *

siStripFEDCheck.RawDataTag = 'rawDataCollector'
siStripFEDMonitor.RawDataTag = 'rawDataCollector'
SiPixelHLTSource.RawInput = 'rawDataCollector'
dqmCSCClient.InputObjects = 'rawDataCollector'
dtDataIntegrityUnpacker.inputLabel = 'rawDataCollector'
hcalMonitor.FEDRawDataCollection = 'rawDataCollector'
hcalDetDiagNoiseMonitor.RawDataLabel = 'rawDataCollector'
hcalRawDataMonitor.FEDRawDataCollection = 'rawDataCollector'
#l1tfed.rawTag = 'rawDataCollector'
ecalMonitorTask.collectionTags.TrigPrimEmulDigi = 'simEcalTriggerPrimitiveDigis'
ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = 'rawDataCollector'
ecalPreshowerRawDataTask.FEDRawDataCollection = 'rawDataCollector'
castorOfflineMonitor.rawLabel = 'rawDataCollector'
cscMonitor.FEDRawDataCollectionTag = 'rawDataCollector'

# L1 Trigger - remove emulator and adapt labels for private unpacking
from DQMOffline.L1Trigger.L1TriggerDqmOfflineMC_cff import *

# put back one of the FastSim aliases, because DQM overwrites it
from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    from FastSimulation.Configuration.DigiAliases_cff import loadTriggerDigiAliases
    loadTriggerDigiAliases()
    from FastSimulation.Configuration.DigiAliases_cff import caloStage1LegacyFormatDigis

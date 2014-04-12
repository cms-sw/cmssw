import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.ClusterTask_cfi import ecalClusterTask
from DQM.EcalBarrelMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import ecalPresampleTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalBarrelMonitorTasks.RecoSummaryTask_cfi import ecalRecoSummaryTask
from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalBarrelMonitorClient.SelectiveReadoutClient_cfi import ecalSelectiveReadoutClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import ecalSummaryClient

ecalMEFormatter = cms.EDAnalyzer("EcalMEFormatter",
    MEs = cms.untracked.PSet(),
    verbosity = cms.untracked.int32(0)
)

def insertIntoMEFormatterMEs(ecalModule, moduleName):
    for name in ecalModule.MEs.parameterNames_():
        setattr(ecalMEFormatter.MEs, moduleName + name, getattr(ecalModule.MEs, name))

insertIntoMEFormatterMEs(ecalClusterTask, 'ClusterTask')
insertIntoMEFormatterMEs(ecalEnergyTask, 'EnergyTask')
insertIntoMEFormatterMEs(ecalIntegrityTask, 'IntegrityTask')
insertIntoMEFormatterMEs(ecalOccupancyTask, 'OccupancyTask')
insertIntoMEFormatterMEs(ecalPresampleTask, 'PresampleTask')
insertIntoMEFormatterMEs(ecalRawDataTask, 'RawDataTask')
insertIntoMEFormatterMEs(ecalRecoSummaryTask, 'RecoSummaryTask')
insertIntoMEFormatterMEs(ecalTrigPrimTask, 'TrigPrimTask')

insertIntoMEFormatterMEs(ecalIntegrityClient, 'IntegrityClient')
insertIntoMEFormatterMEs(ecalOccupancyClient, 'OccupancyClient')
insertIntoMEFormatterMEs(ecalPresampleClient, 'PresampleClient')
insertIntoMEFormatterMEs(ecalRawDataClient, 'RawDataClient')
insertIntoMEFormatterMEs(ecalSummaryClient, 'SummaryClient')

delattr(ecalMEFormatter.MEs, 'TrigPrimTaskEtMaxEmul')
delattr(ecalMEFormatter.MEs, 'TrigPrimTaskEmulMaxIndex')
delattr(ecalMEFormatter.MEs, 'TrigPrimTaskMatchedIndex')
delattr(ecalMEFormatter.MEs, 'TrigPrimTaskEtEmulError')
delattr(ecalMEFormatter.MEs, 'TrigPrimTaskFGEmulError')

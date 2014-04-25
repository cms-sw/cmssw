import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.ClusterTask_cfi import ecalClusterTask
from DQM.EcalMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalMonitorTasks.PresampleTask_cfi import ecalPresampleTask
from DQM.EcalMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalMonitorTasks.RecoSummaryTask_cfi import ecalRecoSummaryTask
from DQM.EcalMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

from DQM.EcalMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalMonitorClient.SelectiveReadoutClient_cfi import ecalSelectiveReadoutClient
from DQM.EcalMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalMonitorClient.SummaryClient_cfi import ecalSummaryClient

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

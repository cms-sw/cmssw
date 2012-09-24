import FWCore.ParameterSet.Config as cms
from DQM.EcalCommon.dqmpset import *
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPnDiodeTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.PNIntegrityClient_cfi import ecalPnIntegrityClient

occupancyTask = dqmpset(ecalOccupancyTask['MEs'])
integrityTask = dqmpset(ecalIntegrityTask['MEs'])
pnDiodeTask = dqmpset(ecalPnDiodeTask['MEs'])
rawDataTask = dqmpset(ecalRawDataTask['MEs'])
integrityClient = dqmpset(ecalIntegrityClient['MEs'])
pnIntegrityClient = dqmpset(ecalPnIntegrityClient['MEs'])

ecalCondDBWriter = cms.EDAnalyzer("EcalCondDBWriter",
    DBName = cms.untracked.string(""),
    hostName = cms.untracked.string(""),
    hostPort = cms.untracked.int32(0),
    userName = cms.untracked.string(""),
    password = cms.untracked.string(""),
    tagName = cms.untracked.string(""),
    location = cms.untracked.string(""),
    runType = cms.untracked.string(""),
    inputRootFiles = cms.untracked.vstring(),
    workerParams = cms.untracked.PSet(
        Integrity = cms.untracked.PSet(
            Quality = integrityClient.Quality,
            Digi = occupancyTask.Digi,
            Gain = integrityTask.Gain,
            ChId = integrityTask.ChId,
            GainSwitch = integrityTask.GainSwitch,
            TowerId = integrityTask.TowerId,
            BlockSize = integrityTask.BlockSize,
            L1AFE = rawDataTask.L1AFE,
            BXFE = rawDataTask.BXFE,
            MEMDigi = pnDiodeTask.Occupancy,
            MEMChId = pnDiodeTask.MEMChId,
            MEMGain = pnDiodeTask.MEMGain,
            PNQuality = pnIntegrityClient.QualitySummary,
            MEMTowerId = pnDiodeTask.MEMTowerId,
            MEMBlockSize = pnDiodeTask.MEMBlockSize
        ),
        Cosmic = cms.untracked.PSet(),
        Laser = cms.untracked.PSet(),
        Pedestal = cms.untracked.PSet(),
        Presample = cms.untracked.PSet(),
        TestPulse = cms.untracked.PSet(),
        BeamCalo = cms.untracked.PSet(),
        BeamHodo = cms.untracked.PSet(),
        TriggerPrimitives = cms.untracked.PSet(),
        Cluster = cms.untracked.PSet(),
        Timing = cms.untracked.PSet(),
        Led = cms.untracked.PSet(),
        RawData = cms.untracked.PSet(),
        Occupancy = cms.untracked.PSet()
    ),
    verbosity = cms.untracked.int32(0)
)
    

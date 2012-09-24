import FWCore.ParameterSet.Config as cms
from DQM.EcalCommon.dqmpset import *
from DQM.EcalBarrelMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import ecalLaserTask
from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ecalLedTask
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import ecalPedestalTask
from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import ecalPresampleTask
from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPnDiodeTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask
from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.LaserClient_cfi import ecalLaserClient
from DQM.EcalBarrelMonitorClient.LedClient_cfi import ecalLedClient
from DQM.EcalBarrelMonitorClient.PedestalClient_cfi import ecalPedestalClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.PNIntegrityClient_cfi import ecalPnIntegrityClient
from DQM.EcalBarrelMonitorClient.TestPulseClient_cfi import ecalTestPulseClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient

energyTask = dqmpset(ecalEnergyTask['MEs'])
integrityTask = dqmpset(ecalIntegrityTask['MEs'])
laserTask = dqmpset(ecalLaserTask['MEs'])
occupancyTask = dqmpset(ecalOccupancyTask['MEs'])
pedestalTask = dqmpset(ecalPedestalTask['MEs'])
presampleTask = dqmpset(ecalPresampleTask['MEs'])
pnDiodeTask = dqmpset(ecalPnDiodeTask['MEs'])
rawDataTask = dqmpset(ecalRawDataTask['MEs'])
testPulseTask = dqmpset(ecalTestPulseTask['MEs'])
timingTask = dqmpset(ecalTimingTask['MEs'])
integrityClient = dqmpset(ecalIntegrityClient['MEs'])
laserClient = dqmpset(ecalLaserClient['MEs'])
pedestalClient = dqmpset(ecalPedestalClient['MEs'])
presampleClient = dqmpset(ecalPresampleClient['MEs'])
pnIntegrityClient = dqmpset(ecalPnIntegrityClient['MEs'])
testPulseClient = dqmpset(ecalTestPulseClient['MEs'])
timingClient = dqmpset(ecalTimingClient['MEs'])

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
        Laser = cms.untracked.PSet(
            Amplitude = laserTask.Amplitude,
            AOverP = laserTask.AOverP,
            Timing = laserTask.Timing,
            Quality = laserClient.Quality,
            PNAmplitude = laserTask.PNAmplitude,
            PNQuality = laserClient.PNQualitySummary,
            PNPedestal = pnDiodeTask.Pedestal
        ),
        Pedestal = cms.untracked.PSet(
            Pedestal = pedestalTask.Pedestal,
            Quality = pedestalClient.Quality,
            PNPedestal = pedestalTask.PNPedestal,
            PNQuality = pedestalClient.PNQualitySummary
        ),
        Presample = cms.untracked.PSet(
            Pedestal = presampleTask.Pedestal,
            Quality = presampleClient.Quality
        ),
        TestPulse = cms.untracked.PSet(
            Amplitude = testPulseTask.Amplitude,
            Shape = testPulseTask.Shape,
            Quality = testPulseClient.Quality,
            PNAmplitude = testPulseTask.PNAmplitude,
            PNPedestal = pnDiodeTask.Pedestal,
            PNQuality = testPulseClient.PNQualitySummary
        ),
        Timing = cms.untracked.PSet(
            Timing = timingTask.TimeMap,
            Quality = timingClient.Quality
        ),
        Led = cms.untracked.PSet(
            Amplitude = ledTask.Amplitude,
            AOverP = ledTask.AOverP,
            Timing = ledTask.Timing,
            Quality = ledClient.Quality,
            PNAmplitude = ledTask.PNAmplitude,
            PNQuality = ledClient.PNQualitySummary,
            PNPedestal = pnDiodeTask.Pedestal
        ),
        Occupancy = cms.untracked.PSet(
            Occupancy = occupancyTask.Digi,
            Energy = energyTask.HitMap
        ),
        laserWavelengths = cms.untracked.vint32(1, 2, 3, 4),
        ledWavelengths = cms.untracked.vint32(1, 2),
        MGPAGains = cms.untracked.vint32(1, 6, 12),
        MGPAGainsPN = cms.untracked.vint32(1, 16)
    ),
    verbosity = cms.untracked.int32(0)
)

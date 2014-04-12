import FWCore.ParameterSet.Config as cms

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

energyTask = ecalEnergyTask.MEs
integrityTask = ecalIntegrityTask.MEs
laserTask = ecalLaserTask.MEs
ledTask = ecalLedTask.MEs
occupancyTask = ecalOccupancyTask.MEs
pedestalTask = ecalPedestalTask.MEs
presampleTask = ecalPresampleTask.MEs
pnDiodeTask = ecalPnDiodeTask.MEs
rawDataTask = ecalRawDataTask.MEs
testPulseTask = ecalTestPulseTask.MEs
timingTask = ecalTimingTask.MEs
integrityClient = ecalIntegrityClient.MEs
laserClient = ecalLaserClient.MEs
ledClient = ecalLedClient.MEs
pedestalClient = ecalPedestalClient.MEs
presampleClient = ecalPresampleClient.MEs
pnIntegrityClient = ecalPnIntegrityClient.MEs
testPulseClient = ecalTestPulseClient.MEs
timingClient = ecalTimingClient.MEs

cosmic = "COSMIC"
beam = "BEAM"
laser = "LASER"
led = "LED"
testpulse = "TEST_PULSE"
pedestal = "PEDESTAL"
pedestalOffset = "PEDESTAL-OFFSET"
physics = "PHYSICS"
halo = "HALO"

# run tags are only used if the DAQ failed to write the RunIOV
# Otherwise existing IOV will be used

ecalCondDBWriter = cms.EDAnalyzer("EcalCondDBWriter",
    DBName = cms.untracked.string(""),
    hostName = cms.untracked.string(""),
    hostPort = cms.untracked.int32(0),
    userName = cms.untracked.string(""),
    password = cms.untracked.string(""),
    location = cms.untracked.string(""), # tag for the run - where was the run taken?
    runType = cms.untracked.string(""), # tag for the run - what is the purpose of the run? 
    runGeneralTag = cms.untracked.string(""), # tag for the run - global or local?
    monRunGeneralTag = cms.untracked.string(""), # identifies the process writing into DB
    inputRootFiles = cms.untracked.vstring(),
    workerParams = cms.untracked.PSet(
        Integrity = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, laser, testpulse, pedestal, pedestalOffset, led, physics),
            source = cms.untracked.PSet(
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
            )
        ),
        Laser = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, laser, physics),
            source = cms.untracked.PSet(
                Amplitude = laserTask.Amplitude,
                AOverP = laserTask.AOverP,
                Timing = laserTask.Timing,
                Quality = laserClient.Quality,
                PNAmplitude = laserTask.PNAmplitude,
                PNQuality = laserClient.PNQualitySummary,
                PNPedestal = pnDiodeTask.Pedestal
            )
        ),
        Pedestal = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(pedestal),
            source = cms.untracked.PSet(
                Pedestal = pedestalTask.Pedestal,
                Quality = pedestalClient.Quality,
                PNPedestal = pedestalTask.PNPedestal,
                PNQuality = pedestalClient.PNQualitySummary
            )
        ),
        Presample = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, laser, testpulse, pedestal, led, physics),
            source = cms.untracked.PSet(
                Pedestal = presampleTask.Pedestal,
                Quality = presampleClient.Quality
            )
        ),
        TestPulse = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, testpulse, physics),
            source = cms.untracked.PSet(
                Amplitude = testPulseTask.Amplitude,
                Shape = testPulseTask.Shape,
                Quality = testPulseClient.Quality,
                PNAmplitude = testPulseTask.PNAmplitude,
                PNPedestal = pnDiodeTask.Pedestal,
                PNQuality = testPulseClient.PNQualitySummary
            )
        ),
        Timing = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(beam, physics),
            source = cms.untracked.PSet(
                Timing = timingTask.TimeMap,
                Quality = timingClient.Quality
            )
        ),
        Led = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, led, physics),
            source = cms.untracked.PSet(
                Amplitude = ledTask.Amplitude,
                AOverP = ledTask.AOverP,
                Timing = ledTask.Timing,
                Quality = ledClient.Quality
# MON_PN_LED1_DAT and MON_PN_LED2_DAT do not exist in the DB as of September 2012                
#                PNAmplitude = ledTask.PNAmplitude,
#                PNQuality = ledClient.PNQualitySummary,
#                PNPedestal = pnDiodeTask.Pedestal
            )
        ),
        Occupancy = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, laser, testpulse, pedestal, pedestalOffset, led, physics),
            source = cms.untracked.PSet(
                Occupancy = occupancyTask.Digi,
                Energy = energyTask.HitMap
            )
        ),
        Summary = cms.untracked.PSet(
            runTypes = cms.untracked.vstring(cosmic, beam, laser, testpulse, pedestal, pedestalOffset, led, physics, halo)
        ),
        laserWavelengths = cms.untracked.vint32(1, 2, 3, 4),
        ledWavelengths = cms.untracked.vint32(1, 2),
        MGPAGains = cms.untracked.vint32(1, 6, 12),
        MGPAGainsPN = cms.untracked.vint32(1, 16)
    ),
    verbosity = cms.untracked.int32(0)
)

import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorClient.PNIntegrityClient_cfi import ecalPNIntegrityClient
from DQM.EcalMonitorClient.LaserClient_cfi import ecalLaserClient
from DQM.EcalMonitorClient.LedClient_cfi import ecalLedClient
from DQM.EcalMonitorClient.TestPulseClient_cfi import ecalTestPulseClient
from DQM.EcalMonitorClient.PedestalClient_cfi import ecalPedestalClient

from DQM.EcalCommon.CommonParams_cfi import *

activeSources = ['Laser', 'Led', 'TestPulse']

ecalCalibrationSummaryClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        laserWavelengths = ecaldqmLaserWavelengths,
        ledWavelengths = ecaldqmLedWavelengths,
        testPulseMGPAGains = ecaldqmMGPAGains,
        testPulseMGPAGainsPN = ecaldqmMGPAGainsPN,
        pedestalMGPAGains = ecaldqmMGPAGains,
        pedestalMGPAGainsPN = ecaldqmMGPAGainsPN,
        activeSources = cms.untracked.vstring(activeSources),
    ),
    sources = cms.untracked.PSet(
        Laser = ecalLaserClient.MEs.QualitySummary,
        LaserPN = ecalLaserClient.MEs.PNQualitySummary,
        Led = ecalLedClient.MEs.QualitySummary,
        LedPN = ecalLedClient.MEs.PNQualitySummary,
        Pedestal = ecalPedestalClient.MEs.QualitySummary,        
        PedestalPN = ecalPedestalClient.MEs.PNQualitySummary,
        PNIntegrity = ecalPNIntegrityClient.MEs.QualitySummary,
        TestPulse = ecalTestPulseClient.MEs.QualitySummary,        
        TestPulsePN = ecalTestPulseClient.MEs.PNQualitySummary
    ),
    MEs = cms.untracked.PSet(
        PNQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s PN global quality'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the calibration data quality for the PN diodes. Channel is red if it is red in any of the Laser 3, Led 1 and 2, Pedestal gain 12, and Test Pulse gain 12 quality summary.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global calibration quality%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the calibration data quality. Channel is red if it is red in any of the Laser 3, Led 1 and 2, Pedestal gain 12, and Test Pulse gain 12 quality summary.')
        )
    )
)

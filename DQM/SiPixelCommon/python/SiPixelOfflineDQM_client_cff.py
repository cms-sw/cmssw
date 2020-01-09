import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#
# This object is used to make changes for different running scenarios
#

#Client:
sipixelEDAClient = DQMEDHarvester("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(False),
    ActionOnRunEnd = cms.untracked.bool(True),
    HighResolutionOccupancy = cms.untracked.bool(False),
    NoiseRateCutValue = cms.untracked.double(-1.),
    NEventsForNoiseCalculation = cms.untracked.int32(100000),
    UseOfflineXMLFile = cms.untracked.bool(True),
    Tier0Flag = cms.untracked.bool(True),
    DoHitEfficiency = cms.untracked.bool(True),
    isUpgrade = cms.untracked.bool(False)
)

#QualityTester
from DQMServices.Core.DQMQualityTester import DQMQualityTester
sipixelQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_tier0_qualitytest.xml'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    label = cms.untracked.string("SiPixelDQMQTests"),
    verboseQT = cms.untracked.bool(False)
)

#Heavy Ion QualityTester
sipixelQTesterHI = sipixelQTester.clone(
    qtList = cms.untracked.FileInPath('DQM/SiPixelMonitorClient/test/sipixel_tier0_qualitytest_heavyions.xml')
)

#DataCertification:
sipixelDaqInfo = DQMEDHarvester("SiPixelDaqInfo")
sipixelDcsInfo = DQMEDHarvester("SiPixelDcsInfo")
sipixelCertification = DQMEDHarvester("SiPixelCertification")

#Predefined Sequences:
PixelOfflineDQMClient = cms.Sequence(sipixelEDAClient)
PixelOfflineDQMClientWithDataCertification = cms.Sequence(sipixelQTester+
                                                          sipixelEDAClient+
                                                          sipixelDaqInfo+
                                                          sipixelDcsInfo+
                                                          sipixelCertification)
PixelOfflineDQMClientNoDataCertification = cms.Sequence(sipixelQTester+
                                                          sipixelEDAClient)
PixelOfflineDQMClientNoDataCertification_cosmics = cms.Sequence(sipixelQTester+
                                                          sipixelEDAClient)

PixelOfflineDQMClientWithDataCertificationHI = cms.Sequence(PixelOfflineDQMClientNoDataCertification)
PixelOfflineDQMClientWithDataCertificationHI.replace(sipixelQTester,sipixelQTesterHI)

# Modify for running with the Phase 1 pixel detector.
from DQM.SiPixelPhase1Config.SiPixelPhase1OfflineDQM_harvesting_cff import *
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith(PixelOfflineDQMClient, siPixelPhase1OfflineDQM_harvesting)
#TODO: properly upgrade these and the others
phase1Pixel.toReplaceWith(PixelOfflineDQMClientNoDataCertification, siPixelPhase1OfflineDQM_harvesting)
phase1Pixel.toReplaceWith(PixelOfflineDQMClientNoDataCertification_cosmics, siPixelPhase1OfflineDQM_harvesting_cosmics)
phase1Pixel.toReplaceWith(PixelOfflineDQMClientWithDataCertification, siPixelPhase1OfflineDQM_harvesting)
phase1Pixel.toReplaceWith(PixelOfflineDQMClientWithDataCertificationHI, siPixelPhase1OfflineDQM_harvesting_hi)

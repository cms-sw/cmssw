import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#Client:
sipixelEDAClientP5 = DQMEDHarvester("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(True), ## do not set to False, otherwise Summary histos not filled!
    ActionOnRunEnd = cms.untracked.bool(True),
    HighResolutionOccupancy = cms.untracked.bool(True),
    NoiseRateCutValue = cms.untracked.double(-1.), ## set negative if no noisy pixel list should be produced
    NEventsForNoiseCalculation = cms.untracked.int32(500),
    UseOfflineXMLFile = cms.untracked.bool(False),
    Tier0Flag = cms.untracked.bool(False),
    DoHitEfficiency = cms.untracked.bool(False)
)

#DataCertification:
sipixelDaqInfo = DQMEDHarvester("SiPixelDaqInfo")
sipixelDcsInfo = DQMEDHarvester("SiPixelDcsInfo")
sipixelCertification = DQMEDHarvester("SiPixelCertification")

#Predefined Sequences:
PixelP5DQMClient = cms.Sequence(sipixelEDAClientP5)
PixelP5DQMClientWithDataCertification = cms.Sequence(sipixelEDAClientP5+
                                                          sipixelDaqInfo+
							  sipixelDcsInfo+
							  sipixelCertification)

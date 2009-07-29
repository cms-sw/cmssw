import FWCore.ParameterSet.Config as cms

#Client:
sipixelEDAClientP5 = cms.EDFilter("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(True),
    ActionOnRunEnd = cms.untracked.bool(True),
    HighResolutionOccupancy = cms.untracked.bool(True),
    NoiseRateCutValue = cms.untracked.double(-1.),
    NEventsForNoiseCalculation = cms.untracked.int32(2000),
    UseOfflineXMLFile = cms.untracked.bool(False),
    Tier0Flag = cms.untracked.bool(False)
)

#DataCertification:
sipixelDaqInfo = cms.EDFilter("SiPixelDaqInfo")
sipixelDcsInfo = cms.EDFilter("SiPixelDcsInfo")
sipixelCertification = cms.EDFilter("SiPixelCertification")

#Predefined Sequences:
PixelP5DQMClient = cms.Sequence(sipixelEDAClientP5)
PixelP5DQMClientWithDataCertification = cms.Sequence(sipixelEDAClientP5+
                                                          sipixelDaqInfo+
							  sipixelDcsInfo+
							  sipixelCertification)

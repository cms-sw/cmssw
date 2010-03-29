import FWCore.ParameterSet.Config as cms

#Client:
sipixelEDAClient = cms.EDAnalyzer("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(False),
    ActionOnRunEnd = cms.untracked.bool(True),
    HighResolutionOccupancy = cms.untracked.bool(False),
    NoiseRateCutValue = cms.untracked.double(-1.),
    NEventsForNoiseCalculation = cms.untracked.int32(100000),
    UseOfflineXMLFile = cms.untracked.bool(True),
    Tier0Flag = cms.untracked.bool(True)
)

#DataCertification:
sipixelDaqInfo = cms.EDAnalyzer("SiPixelDaqInfo")
sipixelDcsInfo = cms.EDAnalyzer("SiPixelDcsInfo")
sipixelCertification = cms.EDAnalyzer("SiPixelCertification")

#Predefined Sequences:
PixelOfflineDQMClient = cms.Sequence(sipixelEDAClient)
PixelOfflineDQMClientWithDataCertification = cms.Sequence(sipixelEDAClient+
                                                          sipixelDaqInfo+
							  sipixelDcsInfo+
							  sipixelCertification)

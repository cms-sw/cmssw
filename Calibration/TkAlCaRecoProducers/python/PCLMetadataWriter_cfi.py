import FWCore.ParameterSet.Config as cms

pclMetadataWriter = cms.EDAnalyzer("PCLMetadataWriter",
                                   recordsToMap = cms.VPSet()
                                   )


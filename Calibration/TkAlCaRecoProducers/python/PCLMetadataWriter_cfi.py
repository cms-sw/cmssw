import FWCore.ParameterSet.Config as cms

pclMetadataWriter = cms.EDAnalyzer("PCLMetadataWriter",
                                   readFromDB = cms.bool(True),
                                   recordsToMap = cms.VPSet()
                                   )


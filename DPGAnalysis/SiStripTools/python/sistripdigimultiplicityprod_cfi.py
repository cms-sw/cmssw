import FWCore.ParameterSet.Config as cms


ssdigimultprod = cms.EDProducer("SiStripDigiMultiplicityProducer",
                                clusterdigiCollection = cms.InputTag("siStripDigis","ZeroSuppressed"),
                                wantedSubDets = cms.VPSet(    
                                    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK")),
                                    cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB")),
                                    cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID")),
                                    cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB")),
                                    cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"))
                                   )
                                )

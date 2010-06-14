import FWCore.ParameterSet.Config as cms

nmaxPerLumi = cms.EDFilter("NMaxPerLumi",
                          nMaxPerLumi = cms.uint32(10)
                          )

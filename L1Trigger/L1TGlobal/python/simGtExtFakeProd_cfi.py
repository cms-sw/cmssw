
import FWCore.ParameterSet.Config as cms

simGtExtFakeProd = cms.EDProducer("L1TExtCondProducer",
                                  bxFirst = cms.int32(-2),
                                  bxLast = cms.int32(2),
                                  setBptxAND = cms.bool(True),
                                  setBptxPlus = cms.bool(True),
                                  setBptxMinus = cms.bool(True),
                                  setBptxOR = cms.bool(True),
)


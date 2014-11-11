
import FWCore.ParameterSet.Config as cms
lumiInfoRunHeader=cms.EDProducer("LumiInfoRunHeaderProducer",
                                 MCFillSchemeFromConfig = cms.bool(False),
                                 MCFillSchemeFromDB = cms.bool(False),
                                 MCBunchSpacing = cms.int32(450),
                            )

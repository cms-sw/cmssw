
import FWCore.ParameterSet.Config as cms
lumiInfoRunHeaderMC=cms.EDProducer("LumiInfoRunHeaderProducer",
                                 MCFillSchemeFromConfig = cms.bool(True),
                                 MCFillSchemeFromDB = cms.bool(False),
                                 MCBunchSpacing = cms.int32(450),
                            )

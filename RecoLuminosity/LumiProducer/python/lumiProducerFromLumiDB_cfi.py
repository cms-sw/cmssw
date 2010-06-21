
import FWCore.ParameterSet.Config as cms
lumiProducer=cms.EDProducer("LumiProducer",
                            connect=cms.string(''),
                            lumiversion=cms.untracked.string('')
                            )

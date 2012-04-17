
import FWCore.ParameterSet.Config as cms
expressLumiProducer=cms.EDProducer("ExpressLumiProducer",
                            connect=cms.string(''),
                            ncacheEntries=cms.untracked.uint32(5)
                            )

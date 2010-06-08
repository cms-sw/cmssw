
import FWCore.ParameterSet.Config as cms
lumiProducer=cms.EDProducer("LumiProducer",
                            connect=cms.string('frontier://LumiProd/CMS_LUMI_PROD'),
                            lumiversion=cms.untracked.string('0001')
                            )


import FWCore.ParameterSet.Config as cms
lumiProducer=cms.EDProducer("LumiProducer",
                            connect=cms.string('oracle://cms_orcoff_prep/cms_lumi_dev_offline'),
                            lumiversion=cms.untracked.string('0001')
                            )

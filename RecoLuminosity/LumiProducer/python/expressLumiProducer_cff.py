import FWCore.ParameterSet.Config as cms
from RecoLuminosity.LumiProducer.expressLumiProducer_cfi import *
LumiDBService=cms.Service('DBService',
                         authPath=cms.untracked.string('/data/cmsdata')
                         )
expressLumiProducer.connect='oracle://cms_orcoff_prod/cms_runtime_logger'

# foo bar baz
# lEEKg0DPrViML
# SQ2Aj1wLiss7w

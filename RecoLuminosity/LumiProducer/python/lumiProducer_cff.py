import FWCore.ParameterSet.Config as cms
from RecoLuminosity.LumiProducer.lumiProducer_cfi import *
LumiDBService=cms.Service('DBService')
lumiProducer.connect='frontier://LumiProd/CMS_LUMI_PROD'
lumiProducer.lumiversion=''

# foo bar baz
# Bq6FHADkCQm10
# kwDUzks19TAaQ

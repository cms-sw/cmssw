import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import * 
GlobalTag.connect = "frontier://(proxyurl=http://frontier.cms:3128)(serverurl=http://frontier.cms:8000/FrontierOnProd)(serverurl=http://frontier.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_CONDITIONS"
GlobalTag.globaltag = "GR_H_V45"
es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

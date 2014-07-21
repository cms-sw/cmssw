import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import * 
GlobalTag.connect = "frontier://(proxyurl=http://frontier.cms:3128)(serverurl=http://frontier.cms:8000/FrontierOnProd)(serverurl=http://frontier.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_GLOBALTAG"
GlobalTag.globaltag = "GR_H_V38::All"
es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

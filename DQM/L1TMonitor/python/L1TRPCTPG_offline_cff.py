import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TRPCTPG_cfi import *
RPCCabling = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/nfshome0/hltpro/cmssw/cfg/')
    ),
    #  using CondDBSetup
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCEMapRcd'),
        tag = cms.string('RPCEMap_v2')
    )),
    #      string connect = "frontier://(serverurl=http://frontier1.cms:8000/FrontierOn)(serverurl=http://frontier2.cms:8000/FrontierOn)(retrieve-ziplevel=0)/CMS_COND_ON_170_RPC"
    connect = cms.string('frontier://Frontier/CMS_COND_ON_170_RPC'),
    siteLocalConfig = cms.untracked.bool(False)
)

rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source")
)

l1trpctpgpath = cms.Path(rpcunpacker*l1trpctpg)


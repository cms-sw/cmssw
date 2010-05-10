import FWCore.ParameterSet.Config as cms

essapvlatency = cms.ESSource("PoolDBESSource",
                         toGet = cms.VPSet(
    cms.PSet(
    record = cms.string("APVLatencyRcd"),
    tag = cms.string("latency_V0")
    )
    ),
                         connect = cms.string(""),
                         timetype = cms.string("runnumber"),
                         DBParameters = cms.PSet(
    authenticationPath = cms.untracked.string("/afs/cern.ch/cms/DB/conddb"),
    messageLevel = cms.untracked.int32(2)
    )
#                         ,BlobStreamerName = cms.untracked.string("TBufferBlobStreamingService")
                         )


import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

tag = 'SiPhase2OuterTrackerLorentzAngle'
suffix = 'v0'

outdb = 'sqlite_file:'+tag+'_'+suffix+'.db'

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string(outdb)

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('Run'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('Run'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPhase2OuterTrackerLorentzAngleRcd'),
        tag = cms.string(tag)
    ))
)

process.LAPayload = cms.EDAnalyzer("SiPhase2OuterTrackerLorentzAngleWriter")
process.LAPayload.tag = cms.string(tag)

process.p = cms.Path(process.LAPayload)


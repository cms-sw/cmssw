import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("TrackerParametersTest")
process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBESSourceGeometry = cms.ESSource("PoolDBESSource",
                               process.CondDBSetup,
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),tag = cms.string('TK_Parameters_Run3_Test02'))
                                                 ),
                                              connect = cms.string('sqlite_file:./myfilerun3.db'))

process.trackerGeometry.applyAlignment = cms.bool(False)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.p1 = cms.Path(process.test)




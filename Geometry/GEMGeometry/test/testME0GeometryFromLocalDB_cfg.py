import FWCore.ParameterSet.Config as cms
process = cms.Process("Demo")
process.load("Geometry.GEMGeometryBuilder.me0GeometryDB_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")

process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string("ME0RecoGeometryRcd"),
                                             tag = cms.string("ME0RECO_Geometry_Test01"),
                                             connect = cms.string("sqlite:./myfile.db")
                                             )
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.test = cms.EDAnalyzer("ME0GeometryAnalyzer")

process.p = cms.Path(process.test)


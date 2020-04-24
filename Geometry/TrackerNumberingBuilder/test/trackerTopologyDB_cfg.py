import FWCore.ParameterSet.Config as cms

process = cms.Process("NumberingTest")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                             tag = cms.string('TKParameters_Geometry_Test01'),
                                             connect = cms.untracked.string("sqlite_file:./myfile.db")
                                             )
                                    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prod = cms.EDAnalyzer("TrackerTopologyAnalyzer");

process.p1 = cms.Path(process.prod)

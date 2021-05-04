import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerParametersTest")

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                             tag = cms.string('TKParameters_Geometry_TagXX'),
                                             connect = cms.string("sqlite_file:./myfile.db")
                                             )
                                    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.p1 = cms.Path(process.test)




import FWCore.ParameterSet.Config as cms
process = cms.Process("test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
  fileNames =
cms.untracked.vstring('file:/home/amraktad/cms/cmsShow31/EED15312-972B-DE11-B7E8-000423D987E0.root')
 )

process.out = cms.OutputModule("PoolOutputModule",
   fileName = cms.untracked.string('/tmp/data.root')
 )
process.outpath = cms.EndPath(process.out)

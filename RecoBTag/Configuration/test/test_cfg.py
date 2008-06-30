import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/08765709-5826-DD11-9CE8-000423D94700.root')
)

process.RECO = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('reco.root')
)

process.p1 = cms.Path(process.btagging)
process.p = cms.EndPath(process.RECO)



import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V5_v7/0004/12C27642-1362-DD11-825B-000423D6A6F4.root')
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/058/738/FE34639D-4273-DD11-8EBC-0019DB29C614.root')
)

process.myFilter = cms.EDFilter("HcalHPDFilter")

process.Out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    ),
    fileName = cms.untracked.string('hpd_filtered.root')
)

process.p = cms.Path(process.myFilter)
process.outpath = cms.EndPath(process.Out)

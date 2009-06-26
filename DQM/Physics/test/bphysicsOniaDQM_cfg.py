import FWCore.ParameterSet.Config as cms

process = cms.Process("oniaDQM")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.dqmSaver.workflow = cms.untracked.string('/workflow/for/mytest')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    "rfio:/castor/cern.ch/user/m/malgeri/TTbar_RAWDEBUG_pre10_1.root"
                           )
                            )

# DQM monitor module for BPhysics: onia resonances
process.oniaAnalyzer = cms.EDAnalyzer("BPhysicsOniaDQM",
                                      MuonCollection = cms.InputTag("muons"),
)

process.p = cms.Path(process.oniaAnalyzer+process.dqmSaver)


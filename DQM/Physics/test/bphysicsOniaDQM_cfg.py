import FWCore.ParameterSet.Config as cms

process = cms.Process("oniaDQM")
process.load("DQM.Physics.bphysicsOniaDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/workflow/for/mytest')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    "rfio:/castor/cern.ch/user/m/malgeri/TTbar_RAWDEBUG_pre10_1.root"
                           )
                            )

process.p = cms.Path(process.bphysicsOniaDQM+process.dqmSaver)


#-------------- EcnaSystemPythoModuleInsert_1 / beginning
import FWCore.ParameterSet.Config as cms

process = cms.Process("ECNA")

#-------------- Message Logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        last_job_INFOS = cms.untracked.PSet(
            extension = cms.untracked.string('txt')
        )
    ),
    suppressInfo = cms.untracked.vstring('ecalEBunpacker')
)
#-------------- EcnaSystemPythoModuleInsert_1 / end

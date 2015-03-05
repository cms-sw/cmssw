import FWCore.ParameterSet.Config as cms

process = cms.Process("PATUPDATE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff")

process.p = cms.Path( process.makePatJetsUpdated )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("patTupleUpdated.root"),
    outputCommands = cms.untracked.vstring('keep *')
    )

process.endpath = cms.EndPath(process.out)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarPileUpMINIAODSIM
process.source = cms.Source("PoolSource",
  fileNames = filesRelValProdTTbarPileUpMINIAODSIM
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

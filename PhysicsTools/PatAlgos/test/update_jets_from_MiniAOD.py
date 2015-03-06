# This configuration is an example that recalibrates the slimmedJets from MiniAOD
# and adds a new userfloat "oldJetMass" to it

import FWCore.ParameterSet.Config as cms

process = cms.Process("PATUPDATE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarPileUpMINIAODSIM
process.source = cms.Source("PoolSource",
  fileNames = filesRelValProdTTbarPileUpMINIAODSIM
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

process.load("PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff")

from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSSoftDropMass
process.oldJetMass = ak8PFJetsCHSSoftDropMass.clone( src = cms.InputTag("slimmedJets"),
  matched = cms.InputTag("slimmedJets") )
process.patJetsUpdated.userData.userFloats.src += ['oldJetMass']

process.p = cms.Path( process.oldJetMass + process.makePatJetsUpdated )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("patTupleUpdated.root"),
    outputCommands = cms.untracked.vstring('keep *')
    )

process.endpath = cms.EndPath(process.out)


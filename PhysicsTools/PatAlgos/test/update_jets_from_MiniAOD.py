# This configuration is an example that recalibrates the slimmedJets from MiniAOD
# and adds a new userfloat "oldJetMass" to it

import FWCore.ParameterSet.Config as cms

process = cms.Process("PATUPDATE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM
process.source = cms.Source("PoolSource",
  fileNames = filesRelValTTbarPileUpMINIAODSIM
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

process.load("PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff")

# An example where the jet energy correction are updated to the GlobalTag given above
# and a usedfloat containing the previous mass of the jet is added
from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSSoftDropMass
process.oldJetMass = ak8PFJetsCHSSoftDropMass.clone(
  src = cms.InputTag("slimmedJets"),
  matched = cms.InputTag("slimmedJets") )
process.patJetsUpdated.userData.userFloats.src += ['oldJetMass']
process.p = cms.Path( process.oldJetMass + process.makePatJetsUpdated )

# An example where the jet correction is undone
process.patJetCorrFactorsUndoJEC = process.patJetCorrFactorsUpdated.clone(
  src = cms.InputTag("patJetsUpdated"),
  levels = [] )
process.patJetsUndoJEC = process.patJetsUpdated.clone(
  jetSource = cms.InputTag("patJetsUpdated"),
  jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsUndoJEC"))
  )
process.patJetsUndoJEC.userData.userFloats.src = []
process.p += cms.Sequence( process.patJetCorrFactorsUndoJEC + process. patJetsUndoJEC )

# An example where the jet correction are reapplied
process.patJetCorrFactorsReapplyJEC = process.patJetCorrFactorsUpdated.clone(
  src = cms.InputTag("patJetsUndoJEC"),
  levels = ['L1FastJet', 
        'L2Relative', 
        'L3Absolute'] )
process.patJetsReapplyJEC = process.patJetsUpdated.clone(
  jetSource = cms.InputTag("patJetsUndoJEC"),
  jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsReapplyJEC"))
  )
process.patJetsReapplyJEC.userData.userFloats.src = []
process.p += cms.Sequence( process.patJetCorrFactorsReapplyJEC + process. patJetsReapplyJEC )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("patTupleUpdated.root"),
    outputCommands = cms.untracked.vstring('keep *')
    )

process.endpath = cms.EndPath(process.out)


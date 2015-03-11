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
process.patJetCorrFactorsUpdated2 = process.patJetCorrFactorsUpdated.clone(
  src = cms.InputTag("patJetsUpdated"),
  levels = [] )
process.patJetsUpdated2 = process.patJetsUpdated.clone(
  jetSource = cms.InputTag("patJetsUpdated"),
  jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsUpdated2"))
  )
process.patJetsUpdated2.userData.userFloats.src = []
process.p += cms.Sequence( process.patJetCorrFactorsUpdated2 + process. patJetsUpdated2 )

# An example where the jet correction are reapplied
process.patJetCorrFactorsUpdated3 = process.patJetCorrFactorsUpdated.clone(
  src = cms.InputTag("patJetsUpdated2"),
  levels = ['L1FastJet', 
        'L2Relative', 
        'L3Absolute'] )
process.patJetsUpdated3 = process.patJetsUpdated.clone(
  jetSource = cms.InputTag("patJetsUpdated2"),
  jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsUpdated3"))
  )
process.patJetsUpdated3.userData.userFloats.src = []
process.p += cms.Sequence( process.patJetCorrFactorsUpdated3 + process. patJetsUpdated3 )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("patTupleUpdated.root"),
    outputCommands = cms.untracked.vstring('keep *')
    )

process.endpath = cms.EndPath(process.out)


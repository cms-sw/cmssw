import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.jetCorrections_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *

patJetCorrFactorsUpdated = patJetCorrFactors.clone(
   src = cms.InputTag("slimmedJets"),
   primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
   )

## for scheduled mode
makePatJetsUpdated = cms.Sequence(
    patJetCorrFactorsUpdated *
    patJetsUpdated
    )

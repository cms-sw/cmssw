import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.jetCorrections_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *

updatedPatJetCorrFactors = patJetCorrFactors.clone(
   src = cms.InputTag("slimmedJets"),
   primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
   )

## for scheduled mode
makeUpdatedPatJets = cms.Sequence(
    updatedPatJetCorrFactors *
    updatedPatJets
    )

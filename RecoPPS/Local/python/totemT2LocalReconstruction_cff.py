import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.totemT2RecHits_cfi import *

totemT2LocalReconstructionTask = cms.Task(
    totemT2RecHits
)
totemT2LocalReconstruction = cms.Sequence(totemT2LocalReconstructionTask)

import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

## for scheduled mode
patJetCorrectionsTask = cms.Task(patJetCorrFactors)
patJetCorrections = cms.Sequence(patJetCorrectionsTask)

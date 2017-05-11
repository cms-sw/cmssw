import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.updateHPSPFTaus_cff import *

patFixedConePFTauDiscrimination = cms.Sequence()

patHPSPFTauDiscriminationTask = cms.Task(updateHPSPFTausTask)
patHPSPFTauDiscrimination = cms.Sequence(patHPSPFTauDiscriminationTask)

patShrinkingConePFTauDiscrimination = cms.Sequence()

patCaloTauDiscrimination = cms.Sequence()

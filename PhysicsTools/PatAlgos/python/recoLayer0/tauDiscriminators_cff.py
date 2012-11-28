import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.updateHPSPFTaus_cff import *

patFixedConePFTauDiscrimination = cms.Sequence()

patHPSPFTauDiscriminationUpdate = cms.Sequence(updateHPSPFTaus)

patShrinkingConePFTauDiscrimination = cms.Sequence()

patCaloTauDiscrimination = cms.Sequence()

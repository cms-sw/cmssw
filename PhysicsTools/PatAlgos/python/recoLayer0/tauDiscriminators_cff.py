import FWCore.ParameterSet.Config as cms

try:
    from RecoTauTag.Configuration.updateHPSPFTaus_cff import *
    patHPSPFTauDiscriminationUpdate = cms.Sequence(updateHPSPFTaus)
except ImportError:
    patHPSPFTauDiscriminationUpdate = cms.Sequence()

patFixedConePFTauDiscrimination = cms.Sequence()

patShrinkingConePFTauDiscrimination = cms.Sequence()

patCaloTauDiscrimination = cms.Sequence()

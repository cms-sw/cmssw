import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

shrinkingConePFTauDecayModeMassExtractor = cms.EDProducer(
    "PFRecoTauDecayModeValueExtractor",
    # tau collection to discriminate
    PFTauProducer = cms.InputTag('shrinkingConePFTauProducer'),
    PFTauDecayModeProducer = cms.InputTag('shrinkingConePFTauDecayModeProducer'),
    # no prediscriminants needed
    Prediscriminants = noPrediscriminants,
    expression = cms.string('mass()'),
)

shrinkingConePFTauDecayModePtExtractor = shrinkingConePFTauDecayModeMassExtractor.clone()
shrinkingConePFTauDecayModePtExtractor.expression = cms.string('pt()')

shrinkingConePFTauDecayModeEtaExtractor = shrinkingConePFTauDecayModeMassExtractor.clone()
shrinkingConePFTauDecayModeEtaExtractor.expression = cms.string('eta()')

shrinkingConePFTauDecayModePhiExtractor = shrinkingConePFTauDecayModeMassExtractor.clone()
shrinkingConePFTauDecayModePhiExtractor.expression = cms.string('phi()')

shrinkingConePFTauDecayModeExtractors = cms.Sequence(
    shrinkingConePFTauDecayModeMassExtractor  +
    shrinkingConePFTauDecayModePtExtractor +
    shrinkingConePFTauDecayModeEtaExtractor +
    shrinkingConePFTauDecayModePhiExtractor 
)

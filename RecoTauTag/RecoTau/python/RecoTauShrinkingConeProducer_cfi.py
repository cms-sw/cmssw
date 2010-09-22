import FWCore.ParameterSet.Config as cms

'''

Configuration for 'shrinkingCone' PFTau Producer

See PFT-08-001 for a description of the algorithm.

'''

_shrinkingConeRecoTausConfig = cms.PSet(
    name = cms.string("shrinkingCone"),
    pfCandSrc = cms.InputTag("particleFlow"),
    plugin = cms.string("RecoTauBuilderConePlugin"),
    leadObjectPt = cms.double(5.0),
    matchingCone = cms.string('0.1'),
    signalConeChargedHadrons = cms.string('min(max(5.0/et(), 0.07), 0.2)'),
    isoConeChargedHadrons = cms.string('0.5'),
    signalConePiZeros = cms.string('0.15'),
    isoConePiZeros = cms.string('0.5'),
    signalConeNeutralHadrons = cms.string('0.15'),
    isoConeNeutralHadrons = cms.string('0.5'),
)

shrinkingConeRecoTaus = cms.EDProducer(
    "RecoTauProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    piZeroSro = cms.InputTag("ak5PFJetsRecoTauPiZeros"),
    builders = cms.VPSet(
        _shrinkingConeRecoTausConfig
    ),
    modifiers = cms.VPSet(),
)

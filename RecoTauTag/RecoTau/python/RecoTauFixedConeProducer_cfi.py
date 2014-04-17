import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.RecoTauShrinkingConeProducer_cfi import \
        shrinkingConeRecoTaus

'''

Configuration for 'shrinkingCone' PFTau Producer

See PFT-08-001 for a description of the algorithm.

'''

_fixedConeRecoTausConfig = cms.PSet(
    name = cms.string("fixedCone"),
    useClosestPV = cms.bool(False),
    qualityCuts = PFTauQualityCuts,
    # If true, consider PFLeptons (e/mu) as charged hadrons.
    usePFLeptons = cms.bool(True),
    pfCandSrc = cms.InputTag("particleFlow"),
    plugin = cms.string("RecoTauBuilderConePlugin"),
    leadObjectPt = cms.double(5.0),
    matchingCone = cms.string('0.3'),
    signalConeChargedHadrons = cms.string('0.07'),
    isoConeChargedHadrons = cms.string('0.4'),
    signalConePiZeros = cms.string('0.15'),
    isoConePiZeros = cms.string('0.4'),
    signalConeNeutralHadrons = cms.string('0.10'),
    isoConeNeutralHadrons = cms.string('0.4'),
    maxSignalConeChargedHadrons = cms.int32(-1) # CV: upper limit on number of signalConeChargedHadrons disabled per default
)

fixedConeRecoTaus = shrinkingConeRecoTaus.clone(
    builders = cms.VPSet(
        _fixedConeRecoTausConfig
    )
)

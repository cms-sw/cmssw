import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
import RecoTauTag.RecoTau.pfTauPrimaryVertexProducer_cfi as _mod

PFTauPrimaryVertexProducer = _mod.pfTauPrimaryVertexProducer.clone(
    #Algorithm: 0 - use tau-jet vertex, 1 - use vertex[0]
    qualityCuts = PFTauQualityCuts,
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
            selectionCut = cms.double(0.5)
        )
    ),
)

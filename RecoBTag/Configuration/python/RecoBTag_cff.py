import FWCore.ParameterSet.Config as cms

# define the b-tag squences for offline reconstruction
from RecoBTag.SoftLepton.softLepton_cff import *
from RecoBTag.ImpactParameter.impactParameter_cff import *
from RecoBTag.SecondaryVertex.secondaryVertex_cff import *
from RecoBTag.Combined.combinedMVA_cff import *
from RecoBTag.CTagging.RecoCTagging_cff import *
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import *

legacyBTagging = cms.Sequence(
    (
      # impact parameters and IP-only algorithms
      impactParameterTagInfos *
      ( trackCountingHighEffBJetTags +
        trackCountingHighPurBJetTags +
        jetProbabilityBJetTags +
        jetBProbabilityBJetTags +

        # SV tag infos depending on IP tag infos, and SV (+IP) based algos
        secondaryVertexTagInfos *
        ( simpleSecondaryVertexHighEffBJetTags +
          simpleSecondaryVertexHighPurBJetTags +
          combinedSecondaryVertexBJetTags
        )
        + inclusiveSecondaryVertexFinderTagInfos *
        combinedInclusiveSecondaryVertexV2BJetTags

        + ghostTrackVertexTagInfos *
        ghostTrackBJetTags
      ) +

      # soft lepton tag infos and algos
      softPFMuonsTagInfos *
      softPFMuonBJetTags
      + softPFElectronsTagInfos *
      softPFElectronBJetTags
    )

    # overall combined taggers
    * combinedMVABJetTags
    + combinedMVAV2BJetTags
)

# new candidate-based fwk, with PF inputs
pfBTagging = cms.Sequence(
    (
      # impact parameters and IP-only algorithms
      pfImpactParameterTagInfos *
      ( pfTrackCountingHighEffBJetTags +
        pfTrackCountingHighPurBJetTags +
        pfJetProbabilityBJetTags +
        pfJetBProbabilityBJetTags +

        # SV tag infos depending on IP tag infos, and SV (+IP) based algos
        pfSecondaryVertexTagInfos *
        ( pfSimpleSecondaryVertexHighEffBJetTags +
          pfSimpleSecondaryVertexHighPurBJetTags +
          pfCombinedSecondaryVertexBJetTags +
          pfCombinedSecondaryVertexV2BJetTags
        )
        + inclusiveCandidateVertexing *
        pfInclusiveSecondaryVertexFinderTagInfos *
        pfCombinedInclusiveSecondaryVertexV2BJetTags

      ) +

      # soft lepton tag infos and algos
      softPFMuonsTagInfos *
      softPFMuonBJetTags
      + softPFElectronsTagInfos *
      softPFElectronBJetTags
    ) *

    # overall combined taggers
    ( #CSV + soft-lepton + jet probability discriminators combined
      pfCombinedMVABJetTags
      + pfCombinedMVAV2BJetTags

      #CSV + soft-lepton variables combined (btagger)
      + pfCombinedSecondaryVertexSoftLeptonBJetTags
    )
)

btagging = cms.Sequence(
    pfBTagging * pfCTagging
)

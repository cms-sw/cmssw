import FWCore.ParameterSet.Config as cms

# define the b-tag squences for offline reconstruction
from RecoBTag.SoftLepton.softLepton_cff import *
from RecoBTag.ImpactParameter.impactParameter_cff import *
from RecoBTag.SecondaryVertex.secondaryVertex_cff import *
from RecoBTau.JetTagComputer.combinedMVA_cff import *

btagging = cms.Sequence(
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
          combinedSecondaryVertexBJetTags + 
          combinedSecondaryVertexMVABJetTags
        ) +
        ghostTrackVertexTagInfos *
          ghostTrackBJetTags
      ) +

      # soft electrons, tag infos and algorithm(s)
      softElectronCands *
#     softPFElectrons *
      softElectronTagInfos * (
#       softElectronBJetTags +
        softElectronByIP3dBJetTags +
        softElectronByPtBJetTags
      ) +

      # soft muon tag infos and algorithms
      softMuonTagInfos *
      ( softMuonBJetTags +
        softMuonByIP3dBJetTags +
        softMuonByPtBJetTags
      )
    )

    # overall combined taggers
    # * combinedMVABJetTags
)

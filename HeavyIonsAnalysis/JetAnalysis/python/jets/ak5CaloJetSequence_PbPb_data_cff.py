

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak5CaloJets"),
    matched = cms.InputTag("ak5HiGenJets"),
    maxDeltaR = 0.5
    )

ak5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak5CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

ak5Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak5CaloJets"),
    payload = "AK5Calo_HI"
    )

ak5CalopatJets = patJets.clone(jetSource = cms.InputTag("ak5CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak5Calocorr")),
                                               genJetMatch = cms.InputTag("ak5Calomatch"),
                                               genPartonMatch = cms.InputTag("ak5Caloparton"),
                                               jetIDMap = cms.InputTag("ak5CaloJetID"),
                                               addBTagInfo         = False,
                                               addTagInfos         = False,
                                               addDiscriminators   = False,
                                               addAssociatedTracks = False,
                                               addJetCharge        = False,
                                               addJetID            = False,
                                               getJetMCFlavour     = False,
                                               addGenPartonMatch   = False,
                                               addGenJetMatch      = False,
                                               embedGenJetMatch    = False,
                                               embedGenPartonMatch = False,
                                               # embedCaloTowers     = False,
                                               # embedPFCandidates = False
				            )

ak5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak5CalopatJets"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

ak5CaloJetSequence_mc = cms.Sequence(
						  ak5Calomatch
                                                  *
                                                  ak5Caloparton
                                                  *
                                                  ak5Calocorr
                                                  *
                                                  ak5CalopatJets
                                                  *
                                                  ak5CaloJetAnalyzer
                                                  )

ak5CaloJetSequence_data = cms.Sequence(ak5Calocorr
                                                    *
                                                    ak5CalopatJets
                                                    *
                                                    ak5CaloJetAnalyzer
                                                    )

ak5CaloJetSequence_jec = cms.Sequence(ak5CaloJetSequence_mc)
ak5CaloJetSequence_mix = cms.Sequence(ak5CaloJetSequence_mc)

ak5CaloJetSequence = cms.Sequence(ak5CaloJetSequence_data)



import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs6CaloJets"),
    matched = cms.InputTag("ak6HiGenJets"),
    maxDeltaR = 0.6
    )

akVs6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs6CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akVs6Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs6CaloJets"),
    payload = "AKVs6Calo_HI"
    )

akVs6CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs6CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs6Calocorr")),
                                               genJetMatch = cms.InputTag("akVs6Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs6Caloparton"),
                                               jetIDMap = cms.InputTag("akVs6CaloJetID"),
                                               addBTagInfo         = False,
                                               addTagInfos         = False,
                                               addDiscriminators   = False,
                                               addAssociatedTracks = False,
                                               addJetCharge        = False,
                                               addJetID            = False,
                                               getJetMCFlavour     = False,
                                               addGenPartonMatch   = True,
                                               addGenJetMatch      = True,
                                               embedGenJetMatch    = True,
                                               embedGenPartonMatch = True,
                                               # embedCaloTowers     = False,
                                               # embedPFCandidates = False
				            )

akVs6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs6CalopatJets"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs6CaloJetSequence_mc = cms.Sequence(
						  akVs6Calomatch
                                                  *
                                                  akVs6Caloparton
                                                  *
                                                  akVs6Calocorr
                                                  *
                                                  akVs6CalopatJets
                                                  *
                                                  akVs6CaloJetAnalyzer
                                                  )

akVs6CaloJetSequence_data = cms.Sequence(akVs6Calocorr
                                                    *
                                                    akVs6CalopatJets
                                                    *
                                                    akVs6CaloJetAnalyzer
                                                    )

akVs6CaloJetSequence_jec = cms.Sequence(akVs6CaloJetSequence_mc)
akVs6CaloJetSequence_mix = cms.Sequence(akVs6CaloJetSequence_mc)

akVs6CaloJetSequence = cms.Sequence(akVs6CaloJetSequence_mix)

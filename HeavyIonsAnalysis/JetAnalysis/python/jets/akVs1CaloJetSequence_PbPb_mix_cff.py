

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets"),
    maxDeltaR = 0.1
    )

akVs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs1CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akVs1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs1CaloJets"),
    payload = "AKVs1Calo_HI"
    )

akVs1CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs1CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs1Calocorr")),
                                               genJetMatch = cms.InputTag("akVs1Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs1Caloparton"),
                                               jetIDMap = cms.InputTag("akVs1CaloJetID"),
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

akVs1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs1CalopatJets"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs1CaloJetSequence_mc = cms.Sequence(
						  akVs1Calomatch
                                                  *
                                                  akVs1Caloparton
                                                  *
                                                  akVs1Calocorr
                                                  *
                                                  akVs1CalopatJets
                                                  *
                                                  akVs1CaloJetAnalyzer
                                                  )

akVs1CaloJetSequence_data = cms.Sequence(akVs1Calocorr
                                                    *
                                                    akVs1CalopatJets
                                                    *
                                                    akVs1CaloJetAnalyzer
                                                    )

akVs1CaloJetSequence_jec = cms.Sequence(akVs1CaloJetSequence_mc)
akVs1CaloJetSequence_mix = cms.Sequence(akVs1CaloJetSequence_mc)

akVs1CaloJetSequence = cms.Sequence(akVs1CaloJetSequence_mix)

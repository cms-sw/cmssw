

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu3CaloJets"),
    matched = cms.InputTag("ak3HiGenJets"),
    maxDeltaR = 0.3
    )

akPu3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu3CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akPu3Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu3CaloJets"),
    payload = "AKPu3Calo_HI"
    )

akPu3CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu3CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu3Calocorr")),
                                               genJetMatch = cms.InputTag("akPu3Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu3Caloparton"),
                                               jetIDMap = cms.InputTag("akPu3CaloJetID"),
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

akPu3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu3CalopatJets"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu3CaloJetSequence_mc = cms.Sequence(
						  akPu3Calomatch
                                                  *
                                                  akPu3Caloparton
                                                  *
                                                  akPu3Calocorr
                                                  *
                                                  akPu3CalopatJets
                                                  *
                                                  akPu3CaloJetAnalyzer
                                                  )

akPu3CaloJetSequence_data = cms.Sequence(akPu3Calocorr
                                                    *
                                                    akPu3CalopatJets
                                                    *
                                                    akPu3CaloJetAnalyzer
                                                    )

akPu3CaloJetSequence_jec = cms.Sequence(akPu3CaloJetSequence_mc)
akPu3CaloJetSequence_mix = cms.Sequence(akPu3CaloJetSequence_mc)

akPu3CaloJetSequence = cms.Sequence(akPu3CaloJetSequence_mix)

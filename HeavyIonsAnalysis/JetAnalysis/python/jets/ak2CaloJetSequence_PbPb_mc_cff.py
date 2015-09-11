

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

ak2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak2CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

ak2Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak2CaloJets"),
    payload = "AK2Calo_HI"
    )

ak2CalopatJets = patJets.clone(jetSource = cms.InputTag("ak2CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak2Calocorr")),
                                               genJetMatch = cms.InputTag("ak2Calomatch"),
                                               genPartonMatch = cms.InputTag("ak2Caloparton"),
                                               jetIDMap = cms.InputTag("ak2CaloJetID"),
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

ak2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak2CalopatJets"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

ak2CaloJetSequence_mc = cms.Sequence(
						  ak2Calomatch
                                                  *
                                                  ak2Caloparton
                                                  *
                                                  ak2Calocorr
                                                  *
                                                  ak2CalopatJets
                                                  *
                                                  ak2CaloJetAnalyzer
                                                  )

ak2CaloJetSequence_data = cms.Sequence(ak2Calocorr
                                                    *
                                                    ak2CalopatJets
                                                    *
                                                    ak2CaloJetAnalyzer
                                                    )

ak2CaloJetSequence_jec = cms.Sequence(ak2CaloJetSequence_mc)
ak2CaloJetSequence_mix = cms.Sequence(ak2CaloJetSequence_mc)

ak2CaloJetSequence = cms.Sequence(ak2CaloJetSequence_mc)



import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets"),
    maxDeltaR = 0.1
    )

ak1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak1CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

ak1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak1CaloJets"),
    payload = "AK1Calo_HI"
    )

ak1CalopatJets = patJets.clone(jetSource = cms.InputTag("ak1CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak1Calocorr")),
                                               genJetMatch = cms.InputTag("ak1Calomatch"),
                                               genPartonMatch = cms.InputTag("ak1Caloparton"),
                                               jetIDMap = cms.InputTag("ak1CaloJetID"),
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

ak1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak1CalopatJets"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

ak1CaloJetSequence_mc = cms.Sequence(
						  ak1Calomatch
                                                  *
                                                  ak1Caloparton
                                                  *
                                                  ak1Calocorr
                                                  *
                                                  ak1CalopatJets
                                                  *
                                                  ak1CaloJetAnalyzer
                                                  )

ak1CaloJetSequence_data = cms.Sequence(ak1Calocorr
                                                    *
                                                    ak1CalopatJets
                                                    *
                                                    ak1CaloJetAnalyzer
                                                    )

ak1CaloJetSequence_jec = cms.Sequence(ak1CaloJetSequence_mc)
ak1CaloJetSequence_mix = cms.Sequence(ak1CaloJetSequence_mc)

ak1CaloJetSequence = cms.Sequence(ak1CaloJetSequence_jec)
ak1CaloJetAnalyzer.genPtMin = cms.untracked.double(1)

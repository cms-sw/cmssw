

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs3PFJets"),
    matched = cms.InputTag("ak3HiGenJets"),
    maxDeltaR = 0.3
    )

akVs3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akVs3PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs3PFJets"),
    payload = "AKVs3PF_generalTracks"
    )

akVs3PFpatJets = patJets.clone(jetSource = cms.InputTag("akVs3PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs3PFcorr")),
                                               genJetMatch = cms.InputTag("akVs3PFmatch"),
                                               genPartonMatch = cms.InputTag("akVs3PFparton"),
                                               jetIDMap = cms.InputTag("akVs3PFJetID"),
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

akVs3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs3PFpatJets"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs3PFJetSequence_mc = cms.Sequence(
						  akVs3PFmatch
                                                  *
                                                  akVs3PFparton
                                                  *
                                                  akVs3PFcorr
                                                  *
                                                  akVs3PFpatJets
                                                  *
                                                  akVs3PFJetAnalyzer
                                                  )

akVs3PFJetSequence_data = cms.Sequence(akVs3PFcorr
                                                    *
                                                    akVs3PFpatJets
                                                    *
                                                    akVs3PFJetAnalyzer
                                                    )

akVs3PFJetSequence_jec = cms.Sequence(akVs3PFJetSequence_mc)
akVs3PFJetSequence_mix = cms.Sequence(akVs3PFJetSequence_mc)

akVs3PFJetSequence = cms.Sequence(akVs3PFJetSequence_mix)

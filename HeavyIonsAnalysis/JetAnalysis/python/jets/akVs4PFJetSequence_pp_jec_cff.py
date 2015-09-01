

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs4PFJets"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

akVs4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akVs4PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs4PFJets"),
    payload = "AKVs4PF_generalTracks"
    )

akVs4PFpatJets = patJets.clone(jetSource = cms.InputTag("akVs4PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs4PFcorr")),
                                               genJetMatch = cms.InputTag("akVs4PFmatch"),
                                               genPartonMatch = cms.InputTag("akVs4PFparton"),
                                               jetIDMap = cms.InputTag("akVs4PFJetID"),
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

akVs4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs4PFpatJets"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs4PFJetSequence_mc = cms.Sequence(
						  akVs4PFmatch
                                                  *
                                                  akVs4PFparton
                                                  *
                                                  akVs4PFcorr
                                                  *
                                                  akVs4PFpatJets
                                                  *
                                                  akVs4PFJetAnalyzer
                                                  )

akVs4PFJetSequence_data = cms.Sequence(akVs4PFcorr
                                                    *
                                                    akVs4PFpatJets
                                                    *
                                                    akVs4PFJetAnalyzer
                                                    )

akVs4PFJetSequence_jec = cms.Sequence(akVs4PFJetSequence_mc)
akVs4PFJetSequence_mix = cms.Sequence(akVs4PFJetSequence_mc)

akVs4PFJetSequence = cms.Sequence(akVs4PFJetSequence_jec)
akVs4PFJetAnalyzer.genPtMin = cms.untracked.double(1)

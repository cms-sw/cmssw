

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak1PFJets"),
    matched = cms.InputTag("ak1HiGenJets"),
    maxDeltaR = 0.1
    )

ak1PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak1PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

ak1PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak1PFJets"),
    payload = "AK1PF_generalTracks"
    )

ak1PFpatJets = patJets.clone(jetSource = cms.InputTag("ak1PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak1PFcorr")),
                                               genJetMatch = cms.InputTag("ak1PFmatch"),
                                               genPartonMatch = cms.InputTag("ak1PFparton"),
                                               jetIDMap = cms.InputTag("ak1PFJetID"),
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

ak1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak1PFpatJets"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

ak1PFJetSequence_mc = cms.Sequence(
						  ak1PFmatch
                                                  *
                                                  ak1PFparton
                                                  *
                                                  ak1PFcorr
                                                  *
                                                  ak1PFpatJets
                                                  *
                                                  ak1PFJetAnalyzer
                                                  )

ak1PFJetSequence_data = cms.Sequence(ak1PFcorr
                                                    *
                                                    ak1PFpatJets
                                                    *
                                                    ak1PFJetAnalyzer
                                                    )

ak1PFJetSequence_jec = cms.Sequence(ak1PFJetSequence_mc)
ak1PFJetSequence_mix = cms.Sequence(ak1PFJetSequence_mc)

ak1PFJetSequence = cms.Sequence(ak1PFJetSequence_data)

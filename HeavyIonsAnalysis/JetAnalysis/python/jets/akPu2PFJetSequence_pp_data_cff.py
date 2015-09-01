

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu2PFJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

akPu2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu2PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akPu2PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu2PFJets"),
    payload = "AKPu2PF_generalTracks"
    )

akPu2PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu2PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu2PFcorr")),
                                               genJetMatch = cms.InputTag("akPu2PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu2PFparton"),
                                               jetIDMap = cms.InputTag("akPu2PFJetID"),
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

akPu2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu2PFpatJets"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu2PFJetSequence_mc = cms.Sequence(
						  akPu2PFmatch
                                                  *
                                                  akPu2PFparton
                                                  *
                                                  akPu2PFcorr
                                                  *
                                                  akPu2PFpatJets
                                                  *
                                                  akPu2PFJetAnalyzer
                                                  )

akPu2PFJetSequence_data = cms.Sequence(akPu2PFcorr
                                                    *
                                                    akPu2PFpatJets
                                                    *
                                                    akPu2PFJetAnalyzer
                                                    )

akPu2PFJetSequence_jec = cms.Sequence(akPu2PFJetSequence_mc)
akPu2PFJetSequence_mix = cms.Sequence(akPu2PFJetSequence_mc)

akPu2PFJetSequence = cms.Sequence(akPu2PFJetSequence_data)

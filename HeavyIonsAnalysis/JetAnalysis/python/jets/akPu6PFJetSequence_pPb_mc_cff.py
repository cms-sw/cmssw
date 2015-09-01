

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu6PFJets"),
    matched = cms.InputTag("ak6HiGenJets"),
    maxDeltaR = 0.6
    )

akPu6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu6PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu6PFJets"),
    payload = "AKPu6PF_generalTracks"
    )

akPu6PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu6PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu6PFcorr")),
                                               genJetMatch = cms.InputTag("akPu6PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu6PFparton"),
                                               jetIDMap = cms.InputTag("akPu6PFJetID"),
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

akPu6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu6PFpatJets"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu6PFJetSequence_mc = cms.Sequence(
						  akPu6PFmatch
                                                  *
                                                  akPu6PFparton
                                                  *
                                                  akPu6PFcorr
                                                  *
                                                  akPu6PFpatJets
                                                  *
                                                  akPu6PFJetAnalyzer
                                                  )

akPu6PFJetSequence_data = cms.Sequence(akPu6PFcorr
                                                    *
                                                    akPu6PFpatJets
                                                    *
                                                    akPu6PFJetAnalyzer
                                                    )

akPu6PFJetSequence_jec = cms.Sequence(akPu6PFJetSequence_mc)
akPu6PFJetSequence_mix = cms.Sequence(akPu6PFJetSequence_mc)

akPu6PFJetSequence = cms.Sequence(akPu6PFJetSequence_mc)

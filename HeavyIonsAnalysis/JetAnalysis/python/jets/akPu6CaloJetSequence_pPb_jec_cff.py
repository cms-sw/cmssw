

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu6CaloJets"),
    matched = cms.InputTag("ak6HiGenJets"),
    maxDeltaR = 0.6
    )

akPu6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu6Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu6CaloJets"),
    payload = "AKPu6Calo_HI"
    )

akPu6CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu6CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu6Calocorr")),
                                               genJetMatch = cms.InputTag("akPu6Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu6Caloparton"),
                                               jetIDMap = cms.InputTag("akPu6CaloJetID"),
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

akPu6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu6CalopatJets"),
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

akPu6CaloJetSequence_mc = cms.Sequence(
						  akPu6Calomatch
                                                  *
                                                  akPu6Caloparton
                                                  *
                                                  akPu6Calocorr
                                                  *
                                                  akPu6CalopatJets
                                                  *
                                                  akPu6CaloJetAnalyzer
                                                  )

akPu6CaloJetSequence_data = cms.Sequence(akPu6Calocorr
                                                    *
                                                    akPu6CalopatJets
                                                    *
                                                    akPu6CaloJetAnalyzer
                                                    )

akPu6CaloJetSequence_jec = cms.Sequence(akPu6CaloJetSequence_mc)
akPu6CaloJetSequence_mix = cms.Sequence(akPu6CaloJetSequence_mc)

akPu6CaloJetSequence = cms.Sequence(akPu6CaloJetSequence_jec)
akPu6CaloJetAnalyzer.genPtMin = cms.untracked.double(1)



import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets"),
    maxDeltaR = 0.1
    )

akPu1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu1CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu1CaloJets"),
    payload = "AKPu1Calo_HI"
    )

akPu1CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu1CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu1Calocorr")),
                                               genJetMatch = cms.InputTag("akPu1Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu1Caloparton"),
                                               jetIDMap = cms.InputTag("akPu1CaloJetID"),
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

akPu1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu1CalopatJets"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu1CaloJetSequence_mc = cms.Sequence(
						  akPu1Calomatch
                                                  *
                                                  akPu1Caloparton
                                                  *
                                                  akPu1Calocorr
                                                  *
                                                  akPu1CalopatJets
                                                  *
                                                  akPu1CaloJetAnalyzer
                                                  )

akPu1CaloJetSequence_data = cms.Sequence(akPu1Calocorr
                                                    *
                                                    akPu1CalopatJets
                                                    *
                                                    akPu1CaloJetAnalyzer
                                                    )

akPu1CaloJetSequence_jec = cms.Sequence(akPu1CaloJetSequence_mc)
akPu1CaloJetSequence_mix = cms.Sequence(akPu1CaloJetSequence_mc)

akPu1CaloJetSequence = cms.Sequence(akPu1CaloJetSequence_jec)
akPu1CaloJetAnalyzer.genPtMin = cms.untracked.double(1)

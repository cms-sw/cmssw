

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu5CaloJets"),
    matched = cms.InputTag("ak5HiGenJets")
    )

akPu5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu5CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akPu5Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu5CaloJets"),
    payload = "AKPu5Calo_HI"
    )

akPu5CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu5CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu5Calocorr")),
                                               genJetMatch = cms.InputTag("akPu5Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu5Caloparton"),
                                               jetIDMap = cms.InputTag("akPu5CaloJetID"),
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
                                               embedCaloTowers     = False,
                                               embedPFCandidates = False
				            )

akPu5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu5CalopatJets"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu5CaloJetSequence_mc = cms.Sequence(
						  akPu5Calomatch
                                                  *
                                                  akPu5Caloparton
                                                  *
                                                  akPu5Calocorr
                                                  *
                                                  akPu5CalopatJets
                                                  *
                                                  akPu5CaloJetAnalyzer
                                                  )

akPu5CaloJetSequence_data = cms.Sequence(akPu5Calocorr
                                                    *
                                                    akPu5CalopatJets
                                                    *
                                                    akPu5CaloJetAnalyzer
                                                    )

akPu5CaloJetSequence_jec = akPu5CaloJetSequence_mc
akPu5CaloJetSequence_mix = akPu5CaloJetSequence_mc

akPu5CaloJetSequence = cms.Sequence(akPu5CaloJetSequence_mc)



import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs3CaloJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

akVs3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs3Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs3CaloJets"),
    payload = "AKVs3Calo_HI"
    )

akVs3CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs3CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs3Calocorr")),
                                               genJetMatch = cms.InputTag("akVs3Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs3Caloparton"),
                                               jetIDMap = cms.InputTag("akVs3CaloJetID"),
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

akVs3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs3CalopatJets"),
                                                             genjetTag = 'ak3HiGenJetsCleaned',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs3CaloJetSequence_mc = cms.Sequence(
						  akVs3Calomatch
                                                  *
                                                  akVs3Caloparton
                                                  *
                                                  akVs3Calocorr
                                                  *
                                                  akVs3CalopatJets
                                                  *
                                                  akVs3CaloJetAnalyzer
                                                  )

akVs3CaloJetSequence_data = cms.Sequence(akVs3Calocorr
                                                    *
                                                    akVs3CalopatJets
                                                    *
                                                    akVs3CaloJetAnalyzer
                                                    )

akVs3CaloJetSequence_jec = akVs3CaloJetSequence_mc
akVs3CaloJetSequence_mix = akVs3CaloJetSequence_mc

akVs3CaloJetSequence = cms.Sequence(akVs3CaloJetSequence_mc)

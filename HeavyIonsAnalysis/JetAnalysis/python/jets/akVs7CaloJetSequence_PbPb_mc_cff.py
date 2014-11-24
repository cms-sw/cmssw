

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs7CaloJets"),
    matched = cms.InputTag("ak7HiGenJetsCleaned")
    )

akVs7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs7CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs7CaloJets"),
    payload = "AKVs7Calo_HI"
    )

akVs7CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs7CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs7Calocorr")),
                                               genJetMatch = cms.InputTag("akVs7Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs7Caloparton"),
                                               jetIDMap = cms.InputTag("akVs7CaloJetID"),
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

akVs7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs7CalopatJets"),
                                                             genjetTag = 'ak7HiGenJetsCleaned',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs7CaloJetSequence_mc = cms.Sequence(
						  akVs7Calomatch
                                                  *
                                                  akVs7Caloparton
                                                  *
                                                  akVs7Calocorr
                                                  *
                                                  akVs7CalopatJets
                                                  *
                                                  akVs7CaloJetAnalyzer
                                                  )

akVs7CaloJetSequence_data = cms.Sequence(akVs7Calocorr
                                                    *
                                                    akVs7CalopatJets
                                                    *
                                                    akVs7CaloJetAnalyzer
                                                    )

akVs7CaloJetSequence_jec = akVs7CaloJetSequence_mc
akVs7CaloJetSequence_mix = akVs7CaloJetSequence_mc

akVs7CaloJetSequence = cms.Sequence(akVs7CaloJetSequence_mc)

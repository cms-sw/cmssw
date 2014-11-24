

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets")
    )

akVs2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akVs2Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs2CaloJets"),
    payload = "AKVs2Calo_HI"
    )

akVs2CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs2CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs2Calocorr")),
                                               genJetMatch = cms.InputTag("akVs2Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs2Caloparton"),
                                               jetIDMap = cms.InputTag("akVs2CaloJetID"),
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
                                               embedCaloTowers     = False,
                                               embedPFCandidates = False
				            )

akVs2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs2CalopatJets"),
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

akVs2CaloJetSequence_mc = cms.Sequence(
						  akVs2Calomatch
                                                  *
                                                  akVs2Caloparton
                                                  *
                                                  akVs2Calocorr
                                                  *
                                                  akVs2CalopatJets
                                                  *
                                                  akVs2CaloJetAnalyzer
                                                  )

akVs2CaloJetSequence_data = cms.Sequence(akVs2Calocorr
                                                    *
                                                    akVs2CalopatJets
                                                    *
                                                    akVs2CaloJetAnalyzer
                                                    )

akVs2CaloJetSequence_jec = akVs2CaloJetSequence_mc
akVs2CaloJetSequence_mix = akVs2CaloJetSequence_mc

akVs2CaloJetSequence = cms.Sequence(akVs2CaloJetSequence_data)

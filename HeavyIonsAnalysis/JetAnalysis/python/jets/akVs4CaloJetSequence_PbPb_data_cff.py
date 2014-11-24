

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs4CaloJets"),
    matched = cms.InputTag("ak4HiGenJetsCleaned")
    )

akVs4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs4Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs4CaloJets"),
    payload = "AKVs4Calo_HI"
    )

akVs4CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs4CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs4Calocorr")),
                                               genJetMatch = cms.InputTag("akVs4Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs4Caloparton"),
                                               jetIDMap = cms.InputTag("akVs4CaloJetID"),
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

akVs4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs4CalopatJets"),
                                                             genjetTag = 'ak4HiGenJetsCleaned',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs4CaloJetSequence_mc = cms.Sequence(
						  akVs4Calomatch
                                                  *
                                                  akVs4Caloparton
                                                  *
                                                  akVs4Calocorr
                                                  *
                                                  akVs4CalopatJets
                                                  *
                                                  akVs4CaloJetAnalyzer
                                                  )

akVs4CaloJetSequence_data = cms.Sequence(akVs4Calocorr
                                                    *
                                                    akVs4CalopatJets
                                                    *
                                                    akVs4CaloJetAnalyzer
                                                    )

akVs4CaloJetSequence_jec = akVs4CaloJetSequence_mc
akVs4CaloJetSequence_mix = akVs4CaloJetSequence_mc

akVs4CaloJetSequence = cms.Sequence(akVs4CaloJetSequence_data)

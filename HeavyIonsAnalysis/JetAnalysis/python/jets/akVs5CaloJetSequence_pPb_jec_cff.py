

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs5CaloJets"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

akVs5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs5Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs5CaloJets"),
    payload = "AKVs5Calo_HI"
    )

akVs5CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs5CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5Calocorr")),
                                               genJetMatch = cms.InputTag("akVs5Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs5Caloparton"),
                                               jetIDMap = cms.InputTag("akVs5CaloJetID"),
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

akVs5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5CalopatJets"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs5CaloJetSequence_mc = cms.Sequence(
						  akVs5Calomatch
                                                  *
                                                  akVs5Caloparton
                                                  *
                                                  akVs5Calocorr
                                                  *
                                                  akVs5CalopatJets
                                                  *
                                                  akVs5CaloJetAnalyzer
                                                  )

akVs5CaloJetSequence_data = cms.Sequence(akVs5Calocorr
                                                    *
                                                    akVs5CalopatJets
                                                    *
                                                    akVs5CaloJetAnalyzer
                                                    )

akVs5CaloJetSequence_jec = akVs5CaloJetSequence_mc
akVs5CaloJetSequence_mix = akVs5CaloJetSequence_mc

akVs5CaloJetSequence = cms.Sequence(akVs5CaloJetSequence_jec)
akVs5CaloJetAnalyzer.genPtMin = cms.untracked.double(1)

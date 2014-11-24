

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs1CaloJets"),
    matched = cms.InputTag("ak1HiGenJetsCleaned")
    )

akVs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs1CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs1CaloJets"),
    payload = "AKVs1Calo_HI"
    )

akVs1CalopatJets = patJets.clone(jetSource = cms.InputTag("akVs1CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs1Calocorr")),
                                               genJetMatch = cms.InputTag("akVs1Calomatch"),
                                               genPartonMatch = cms.InputTag("akVs1Caloparton"),
                                               jetIDMap = cms.InputTag("akVs1CaloJetID"),
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

akVs1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs1CalopatJets"),
                                                             genjetTag = 'ak1HiGenJetsCleaned',
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

akVs1CaloJetSequence_mc = cms.Sequence(
						  akVs1Calomatch
                                                  *
                                                  akVs1Caloparton
                                                  *
                                                  akVs1Calocorr
                                                  *
                                                  akVs1CalopatJets
                                                  *
                                                  akVs1CaloJetAnalyzer
                                                  )

akVs1CaloJetSequence_data = cms.Sequence(akVs1Calocorr
                                                    *
                                                    akVs1CalopatJets
                                                    *
                                                    akVs1CaloJetAnalyzer
                                                    )

akVs1CaloJetSequence_jec = akVs1CaloJetSequence_mc
akVs1CaloJetSequence_mix = akVs1CaloJetSequence_mc

akVs1CaloJetSequence = cms.Sequence(akVs1CaloJetSequence_jec)
akVs1CaloJetAnalyzer.genPtMin = cms.untracked.double(1)

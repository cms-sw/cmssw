

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets")
    )

akPu2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu2CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akPu2Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu2CaloJets"),
    payload = "AKPu2Calo_HI"
    )

akPu2CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu2CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu2Calocorr")),
                                               genJetMatch = cms.InputTag("akPu2Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu2Caloparton"),
                                               jetIDMap = cms.InputTag("akPu2CaloJetID"),
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

akPu2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu2CalopatJets"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu2CaloJetSequence_mc = cms.Sequence(
						  akPu2Calomatch
                                                  *
                                                  akPu2Caloparton
                                                  *
                                                  akPu2Calocorr
                                                  *
                                                  akPu2CalopatJets
                                                  *
                                                  akPu2CaloJetAnalyzer
                                                  )

akPu2CaloJetSequence_data = cms.Sequence(akPu2Calocorr
                                                    *
                                                    akPu2CalopatJets
                                                    *
                                                    akPu2CaloJetAnalyzer
                                                    )

akPu2CaloJetSequence_jec = akPu2CaloJetSequence_mc
akPu2CaloJetSequence_mix = akPu2CaloJetSequence_mc

akPu2CaloJetSequence = cms.Sequence(akPu2CaloJetSequence_mc)

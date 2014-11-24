

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu7CaloJets"),
    matched = cms.InputTag("ak7HiGenJets")
    )

akPu7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu7CaloJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akPu7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu7CaloJets"),
    payload = "AKPu7Calo_HI"
    )

akPu7CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu7CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu7Calocorr")),
                                               genJetMatch = cms.InputTag("akPu7Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu7Caloparton"),
                                               jetIDMap = cms.InputTag("akPu7CaloJetID"),
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

akPu7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu7CalopatJets"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu7CaloJetSequence_mc = cms.Sequence(
						  akPu7Calomatch
                                                  *
                                                  akPu7Caloparton
                                                  *
                                                  akPu7Calocorr
                                                  *
                                                  akPu7CalopatJets
                                                  *
                                                  akPu7CaloJetAnalyzer
                                                  )

akPu7CaloJetSequence_data = cms.Sequence(akPu7Calocorr
                                                    *
                                                    akPu7CalopatJets
                                                    *
                                                    akPu7CaloJetAnalyzer
                                                    )

akPu7CaloJetSequence_jec = akPu7CaloJetSequence_mc
akPu7CaloJetSequence_mix = akPu7CaloJetSequence_mc

akPu7CaloJetSequence = cms.Sequence(akPu7CaloJetSequence_jec)
akPu7CaloJetAnalyzer.genPtMin = cms.untracked.double(1)

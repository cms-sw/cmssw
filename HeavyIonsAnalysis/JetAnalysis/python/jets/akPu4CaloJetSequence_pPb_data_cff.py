

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu4CaloJets"),
    matched = cms.InputTag("ak4HiGenJetsCleaned")
    )

akPu4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu4CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu4Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu4CaloJets"),
    payload = "AKPu4Calo_HI"
    )

akPu4CalopatJets = patJets.clone(jetSource = cms.InputTag("akPu4CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu4Calocorr")),
                                               genJetMatch = cms.InputTag("akPu4Calomatch"),
                                               genPartonMatch = cms.InputTag("akPu4Caloparton"),
                                               jetIDMap = cms.InputTag("akPu4CaloJetID"),
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

akPu4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu4CalopatJets"),
                                                             genjetTag = 'ak4HiGenJetsCleaned',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu4CaloJetSequence_mc = cms.Sequence(
						  akPu4Calomatch
                                                  *
                                                  akPu4Caloparton
                                                  *
                                                  akPu4Calocorr
                                                  *
                                                  akPu4CalopatJets
                                                  *
                                                  akPu4CaloJetAnalyzer
                                                  )

akPu4CaloJetSequence_data = cms.Sequence(akPu4Calocorr
                                                    *
                                                    akPu4CalopatJets
                                                    *
                                                    akPu4CaloJetAnalyzer
                                                    )

akPu4CaloJetSequence_jec = akPu4CaloJetSequence_mc
akPu4CaloJetSequence_mix = akPu4CaloJetSequence_mc

akPu4CaloJetSequence = cms.Sequence(akPu4CaloJetSequence_data)

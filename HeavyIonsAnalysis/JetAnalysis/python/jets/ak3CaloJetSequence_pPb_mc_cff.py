

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak3CaloJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

ak3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak3CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

ak3Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("ak3CaloJets"),
    payload = "AK3Calo_HI"
    )

ak3CalopatJets = patJets.clone(jetSource = cms.InputTag("ak3CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak3Calocorr")),
                                               genJetMatch = cms.InputTag("ak3Calomatch"),
                                               genPartonMatch = cms.InputTag("ak3Caloparton"),
                                               jetIDMap = cms.InputTag("ak3CaloJetID"),
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

ak3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak3CalopatJets"),
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

ak3CaloJetSequence_mc = cms.Sequence(
						  ak3Calomatch
                                                  *
                                                  ak3Caloparton
                                                  *
                                                  ak3Calocorr
                                                  *
                                                  ak3CalopatJets
                                                  *
                                                  ak3CaloJetAnalyzer
                                                  )

ak3CaloJetSequence_data = cms.Sequence(ak3Calocorr
                                                    *
                                                    ak3CalopatJets
                                                    *
                                                    ak3CaloJetAnalyzer
                                                    )

ak3CaloJetSequence_jec = ak3CaloJetSequence_mc
ak3CaloJetSequence_mix = ak3CaloJetSequence_mc

ak3CaloJetSequence = cms.Sequence(ak3CaloJetSequence_mc)

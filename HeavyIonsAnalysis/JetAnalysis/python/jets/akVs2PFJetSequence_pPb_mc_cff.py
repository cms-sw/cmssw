

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs2PFJets"),
    matched = cms.InputTag("ak2HiGenJetsCleaned")
    )

akVs2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs2PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs2PFJets"),
    payload = "AKVs2PF_generalTracks"
    )

akVs2PFpatJets = patJets.clone(jetSource = cms.InputTag("akVs2PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs2PFcorr")),
                                               genJetMatch = cms.InputTag("akVs2PFmatch"),
                                               genPartonMatch = cms.InputTag("akVs2PFparton"),
                                               jetIDMap = cms.InputTag("akVs2PFJetID"),
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

akVs2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs2PFpatJets"),
                                                             genjetTag = 'ak2HiGenJetsCleaned',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs2PFJetSequence_mc = cms.Sequence(
						  akVs2PFmatch
                                                  *
                                                  akVs2PFparton
                                                  *
                                                  akVs2PFcorr
                                                  *
                                                  akVs2PFpatJets
                                                  *
                                                  akVs2PFJetAnalyzer
                                                  )

akVs2PFJetSequence_data = cms.Sequence(akVs2PFcorr
                                                    *
                                                    akVs2PFpatJets
                                                    *
                                                    akVs2PFJetAnalyzer
                                                    )

akVs2PFJetSequence_jec = akVs2PFJetSequence_mc
akVs2PFJetSequence_mix = akVs2PFJetSequence_mc

akVs2PFJetSequence = cms.Sequence(akVs2PFJetSequence_mc)

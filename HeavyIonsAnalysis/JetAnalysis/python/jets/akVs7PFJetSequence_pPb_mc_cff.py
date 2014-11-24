

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs7PFJets"),
    matched = cms.InputTag("ak7HiGenJetsCleaned")
    )

akVs7PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs7PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs7PFJets"),
    payload = "AKVs7PF_generalTracks"
    )

akVs7PFpatJets = patJets.clone(jetSource = cms.InputTag("akVs7PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs7PFcorr")),
                                               genJetMatch = cms.InputTag("akVs7PFmatch"),
                                               genPartonMatch = cms.InputTag("akVs7PFparton"),
                                               jetIDMap = cms.InputTag("akVs7PFJetID"),
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

akVs7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs7PFpatJets"),
                                                             genjetTag = 'ak7HiGenJetsCleaned',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs7PFJetSequence_mc = cms.Sequence(
						  akVs7PFmatch
                                                  *
                                                  akVs7PFparton
                                                  *
                                                  akVs7PFcorr
                                                  *
                                                  akVs7PFpatJets
                                                  *
                                                  akVs7PFJetAnalyzer
                                                  )

akVs7PFJetSequence_data = cms.Sequence(akVs7PFcorr
                                                    *
                                                    akVs7PFpatJets
                                                    *
                                                    akVs7PFJetAnalyzer
                                                    )

akVs7PFJetSequence_jec = akVs7PFJetSequence_mc
akVs7PFJetSequence_mix = akVs7PFJetSequence_mc

akVs7PFJetSequence = cms.Sequence(akVs7PFJetSequence_mc)

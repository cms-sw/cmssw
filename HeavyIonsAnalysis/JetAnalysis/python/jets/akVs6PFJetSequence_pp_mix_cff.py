

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs6PFJets"),
    matched = cms.InputTag("ak6HiGenJets")
    )

akVs6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs6PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akVs6PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs6PFJets"),
    payload = "AKVs6PF_generalTracks"
    )

akVs6PFpatJets = patJets.clone(jetSource = cms.InputTag("akVs6PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs6PFcorr")),
                                               genJetMatch = cms.InputTag("akVs6PFmatch"),
                                               genPartonMatch = cms.InputTag("akVs6PFparton"),
                                               jetIDMap = cms.InputTag("akVs6PFJetID"),
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

akVs6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs6PFpatJets"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("hiSignal")
                                                             )

akVs6PFJetSequence_mc = cms.Sequence(
						  akVs6PFmatch
                                                  *
                                                  akVs6PFparton
                                                  *
                                                  akVs6PFcorr
                                                  *
                                                  akVs6PFpatJets
                                                  *
                                                  akVs6PFJetAnalyzer
                                                  )

akVs6PFJetSequence_data = cms.Sequence(akVs6PFcorr
                                                    *
                                                    akVs6PFpatJets
                                                    *
                                                    akVs6PFJetAnalyzer
                                                    )

akVs6PFJetSequence_jec = akVs6PFJetSequence_mc
akVs6PFJetSequence_mix = akVs6PFJetSequence_mc

akVs6PFJetSequence = cms.Sequence(akVs6PFJetSequence_mix)

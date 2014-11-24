

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu3PFJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

akPu3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu3PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu3PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu3PFJets"),
    payload = "AKPu3PF_generalTracks"
    )

akPu3PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu3PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu3PFcorr")),
                                               genJetMatch = cms.InputTag("akPu3PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu3PFparton"),
                                               jetIDMap = cms.InputTag("akPu3PFJetID"),
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

akPu3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu3PFpatJets"),
                                                             genjetTag = 'ak3HiGenJetsCleaned',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu3PFJetSequence_mc = cms.Sequence(
						  akPu3PFmatch
                                                  *
                                                  akPu3PFparton
                                                  *
                                                  akPu3PFcorr
                                                  *
                                                  akPu3PFpatJets
                                                  *
                                                  akPu3PFJetAnalyzer
                                                  )

akPu3PFJetSequence_data = cms.Sequence(akPu3PFcorr
                                                    *
                                                    akPu3PFpatJets
                                                    *
                                                    akPu3PFJetAnalyzer
                                                    )

akPu3PFJetSequence_jec = akPu3PFJetSequence_mc
akPu3PFJetSequence_mix = akPu3PFJetSequence_mc

akPu3PFJetSequence = cms.Sequence(akPu3PFJetSequence_data)

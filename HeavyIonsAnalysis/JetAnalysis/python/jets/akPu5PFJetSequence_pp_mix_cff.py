

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu5PFJets"),
    matched = cms.InputTag("ak5HiGenJets")
    )

akPu5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu5PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

akPu5PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu5PFJets"),
    payload = "AKPu5PF_generalTracks"
    )

akPu5PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu5PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu5PFcorr")),
                                               genJetMatch = cms.InputTag("akPu5PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu5PFparton"),
                                               jetIDMap = cms.InputTag("akPu5PFJetID"),
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

akPu5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu5PFpatJets"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("hiSignal")
                                                             )

akPu5PFJetSequence_mc = cms.Sequence(
						  akPu5PFmatch
                                                  *
                                                  akPu5PFparton
                                                  *
                                                  akPu5PFcorr
                                                  *
                                                  akPu5PFpatJets
                                                  *
                                                  akPu5PFJetAnalyzer
                                                  )

akPu5PFJetSequence_data = cms.Sequence(akPu5PFcorr
                                                    *
                                                    akPu5PFpatJets
                                                    *
                                                    akPu5PFJetAnalyzer
                                                    )

akPu5PFJetSequence_jec = akPu5PFJetSequence_mc
akPu5PFJetSequence_mix = akPu5PFJetSequence_mc

akPu5PFJetSequence = cms.Sequence(akPu5PFJetSequence_mix)

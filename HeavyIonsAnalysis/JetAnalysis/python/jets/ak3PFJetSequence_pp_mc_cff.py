

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak3PFJets"),
    matched = cms.InputTag("ak3HiGenJets")
    )

ak3PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak3PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

ak3PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("ak3PFJets"),
    payload = "AK3PF_generalTracks"
    )

ak3PFpatJets = patJets.clone(jetSource = cms.InputTag("ak3PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak3PFcorr")),
                                               genJetMatch = cms.InputTag("ak3PFmatch"),
                                               genPartonMatch = cms.InputTag("ak3PFparton"),
                                               jetIDMap = cms.InputTag("ak3PFJetID"),
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

ak3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak3PFpatJets"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

ak3PFJetSequence_mc = cms.Sequence(
						  ak3PFmatch
                                                  *
                                                  ak3PFparton
                                                  *
                                                  ak3PFcorr
                                                  *
                                                  ak3PFpatJets
                                                  *
                                                  ak3PFJetAnalyzer
                                                  )

ak3PFJetSequence_data = cms.Sequence(ak3PFcorr
                                                    *
                                                    ak3PFpatJets
                                                    *
                                                    ak3PFJetAnalyzer
                                                    )

ak3PFJetSequence_jec = ak3PFJetSequence_mc
ak3PFJetSequence_mix = ak3PFJetSequence_mc

ak3PFJetSequence = cms.Sequence(ak3PFJetSequence_mc)

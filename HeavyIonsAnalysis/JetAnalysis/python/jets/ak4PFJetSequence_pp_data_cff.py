

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4PFJets"),
    matched = cms.InputTag("ak4HiGenJets")
    )

ak4PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak4PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

ak4PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("ak4PFJets"),
    payload = "AK4PF_generalTracks"
    )

ak4PFpatJets = patJets.clone(jetSource = cms.InputTag("ak4PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak4PFcorr")),
                                               genJetMatch = cms.InputTag("ak4PFmatch"),
                                               genPartonMatch = cms.InputTag("ak4PFparton"),
                                               jetIDMap = cms.InputTag("ak4PFJetID"),
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

ak4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak4PFpatJets"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

ak4PFJetSequence_mc = cms.Sequence(
						  ak4PFmatch
                                                  *
                                                  ak4PFparton
                                                  *
                                                  ak4PFcorr
                                                  *
                                                  ak4PFpatJets
                                                  *
                                                  ak4PFJetAnalyzer
                                                  )

ak4PFJetSequence_data = cms.Sequence(ak4PFcorr
                                                    *
                                                    ak4PFpatJets
                                                    *
                                                    ak4PFJetAnalyzer
                                                    )

ak4PFJetSequence_jec = ak4PFJetSequence_mc
ak4PFJetSequence_mix = ak4PFJetSequence_mc

ak4PFJetSequence = cms.Sequence(ak4PFJetSequence_data)

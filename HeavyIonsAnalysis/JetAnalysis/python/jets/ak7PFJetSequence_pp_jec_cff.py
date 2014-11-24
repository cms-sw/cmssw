

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak7PFJets"),
    matched = cms.InputTag("ak7HiGenJets")
    )

ak7PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak7PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

ak7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("ak7PFJets"),
    payload = "AK7PF_generalTracks"
    )

ak7PFpatJets = patJets.clone(jetSource = cms.InputTag("ak7PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak7PFcorr")),
                                               genJetMatch = cms.InputTag("ak7PFmatch"),
                                               genPartonMatch = cms.InputTag("ak7PFparton"),
                                               jetIDMap = cms.InputTag("ak7PFJetID"),
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

ak7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak7PFpatJets"),
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

ak7PFJetSequence_mc = cms.Sequence(
						  ak7PFmatch
                                                  *
                                                  ak7PFparton
                                                  *
                                                  ak7PFcorr
                                                  *
                                                  ak7PFpatJets
                                                  *
                                                  ak7PFJetAnalyzer
                                                  )

ak7PFJetSequence_data = cms.Sequence(ak7PFcorr
                                                    *
                                                    ak7PFpatJets
                                                    *
                                                    ak7PFJetAnalyzer
                                                    )

ak7PFJetSequence_jec = ak7PFJetSequence_mc
ak7PFJetSequence_mix = ak7PFJetSequence_mc

ak7PFJetSequence = cms.Sequence(ak7PFJetSequence_jec)
ak7PFJetAnalyzer.genPtMin = cms.untracked.double(1)

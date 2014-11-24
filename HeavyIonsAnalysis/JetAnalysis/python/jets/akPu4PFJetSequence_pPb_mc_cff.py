

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu4PFJets"),
    matched = cms.InputTag("ak4HiGenJetsCleaned")
    )

akPu4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu4PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu4PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu4PFJets"),
    payload = "AKPu4PF_generalTracks"
    )

akPu4PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu4PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu4PFcorr")),
                                               genJetMatch = cms.InputTag("akPu4PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu4PFparton"),
                                               jetIDMap = cms.InputTag("akPu4PFJetID"),
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

akPu4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu4PFpatJets"),
                                                             genjetTag = 'ak4HiGenJetsCleaned',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akPu4PFJetSequence_mc = cms.Sequence(
						  akPu4PFmatch
                                                  *
                                                  akPu4PFparton
                                                  *
                                                  akPu4PFcorr
                                                  *
                                                  akPu4PFpatJets
                                                  *
                                                  akPu4PFJetAnalyzer
                                                  )

akPu4PFJetSequence_data = cms.Sequence(akPu4PFcorr
                                                    *
                                                    akPu4PFpatJets
                                                    *
                                                    akPu4PFJetAnalyzer
                                                    )

akPu4PFJetSequence_jec = akPu4PFJetSequence_mc
akPu4PFJetSequence_mix = akPu4PFJetSequence_mc

akPu4PFJetSequence = cms.Sequence(akPu4PFJetSequence_mc)



import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu7PFJets"),
    matched = cms.InputTag("ak7HiGenJetsCleaned")
    )

akPu7PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu7PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu7PFJets"),
    payload = "AKPu7PF_generalTracks"
    )

akPu7PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu7PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu7PFcorr")),
                                               genJetMatch = cms.InputTag("akPu7PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu7PFparton"),
                                               jetIDMap = cms.InputTag("akPu7PFJetID"),
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

akPu7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu7PFpatJets"),
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

akPu7PFJetSequence_mc = cms.Sequence(
						  akPu7PFmatch
                                                  *
                                                  akPu7PFparton
                                                  *
                                                  akPu7PFcorr
                                                  *
                                                  akPu7PFpatJets
                                                  *
                                                  akPu7PFJetAnalyzer
                                                  )

akPu7PFJetSequence_data = cms.Sequence(akPu7PFcorr
                                                    *
                                                    akPu7PFpatJets
                                                    *
                                                    akPu7PFJetAnalyzer
                                                    )

akPu7PFJetSequence_jec = akPu7PFJetSequence_mc
akPu7PFJetSequence_mix = akPu7PFJetSequence_mc

akPu7PFJetSequence = cms.Sequence(akPu7PFJetSequence_mc)

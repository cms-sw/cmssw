

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akPu1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu1PFJets"),
    matched = cms.InputTag("ak1HiGenJetsCleaned")
    )

akPu1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu1PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akPu1PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akPu1PFJets"),
    payload = "AKPu1PF_hiIterativeTracks"
    )

akPu1PFpatJets = patJets.clone(jetSource = cms.InputTag("akPu1PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu1PFcorr")),
                                               genJetMatch = cms.InputTag("akPu1PFmatch"),
                                               genPartonMatch = cms.InputTag("akPu1PFparton"),
                                               jetIDMap = cms.InputTag("akPu1PFJetID"),
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

akPu1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu1PFpatJets"),
                                                             genjetTag = 'ak1HiGenJetsCleaned',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal")
                                                             )

akPu1PFJetSequence_mc = cms.Sequence(
						  akPu1PFmatch
                                                  *
                                                  akPu1PFparton
                                                  *
                                                  akPu1PFcorr
                                                  *
                                                  akPu1PFpatJets
                                                  *
                                                  akPu1PFJetAnalyzer
                                                  )

akPu1PFJetSequence_data = cms.Sequence(akPu1PFcorr
                                                    *
                                                    akPu1PFpatJets
                                                    *
                                                    akPu1PFJetAnalyzer
                                                    )

akPu1PFJetSequence_jec = akPu1PFJetSequence_mc
akPu1PFJetSequence_mix = akPu1PFJetSequence_mc

akPu1PFJetSequence = cms.Sequence(akPu1PFJetSequence_mix)

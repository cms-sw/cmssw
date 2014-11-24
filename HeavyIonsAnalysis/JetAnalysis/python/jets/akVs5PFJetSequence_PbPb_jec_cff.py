

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

akVs5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs5PFJets"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

akVs5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5PFJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

akVs5PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("akVs5PFJets"),
    payload = "AKVs5PF_hiIterativeTracks"
    )

akVs5PFpatJets = patJets.clone(jetSource = cms.InputTag("akVs5PFJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5PFcorr")),
                                               genJetMatch = cms.InputTag("akVs5PFmatch"),
                                               genPartonMatch = cms.InputTag("akVs5PFparton"),
                                               jetIDMap = cms.InputTag("akVs5PFJetID"),
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

akVs5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5PFpatJets"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator")
                                                             )

akVs5PFJetSequence_mc = cms.Sequence(
						  akVs5PFmatch
                                                  *
                                                  akVs5PFparton
                                                  *
                                                  akVs5PFcorr
                                                  *
                                                  akVs5PFpatJets
                                                  *
                                                  akVs5PFJetAnalyzer
                                                  )

akVs5PFJetSequence_data = cms.Sequence(akVs5PFcorr
                                                    *
                                                    akVs5PFpatJets
                                                    *
                                                    akVs5PFJetAnalyzer
                                                    )

akVs5PFJetSequence_jec = akVs5PFJetSequence_mc
akVs5PFJetSequence_mix = akVs5PFJetSequence_mc

akVs5PFJetSequence = cms.Sequence(akVs5PFJetSequence_jec)
akVs5PFJetAnalyzer.genPtMin = cms.untracked.double(1)

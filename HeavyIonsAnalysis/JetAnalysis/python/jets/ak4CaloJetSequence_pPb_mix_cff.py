

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

ak4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4CaloJets"),
    matched = cms.InputTag("ak4HiGenJetsCleaned")
    )

ak4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak4CaloJets"),
                                                        matched = cms.InputTag("hiGenParticles")
                                                        )

ak4Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),                                                                
    src = cms.InputTag("ak4CaloJets"),
    payload = "AK4Calo_HI"
    )

ak4CalopatJets = patJets.clone(jetSource = cms.InputTag("ak4CaloJets"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak4Calocorr")),
                                               genJetMatch = cms.InputTag("ak4Calomatch"),
                                               genPartonMatch = cms.InputTag("ak4Caloparton"),
                                               jetIDMap = cms.InputTag("ak4CaloJetID"),
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

ak4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak4CalopatJets"),
                                                             genjetTag = 'ak4HiGenJetsCleaned',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJets',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal")
                                                             )

ak4CaloJetSequence_mc = cms.Sequence(
						  ak4Calomatch
                                                  *
                                                  ak4Caloparton
                                                  *
                                                  ak4Calocorr
                                                  *
                                                  ak4CalopatJets
                                                  *
                                                  ak4CaloJetAnalyzer
                                                  )

ak4CaloJetSequence_data = cms.Sequence(ak4Calocorr
                                                    *
                                                    ak4CalopatJets
                                                    *
                                                    ak4CaloJetAnalyzer
                                                    )

ak4CaloJetSequence_jec = ak4CaloJetSequence_mc
ak4CaloJetSequence_mix = ak4CaloJetSequence_mc

ak4CaloJetSequence = cms.Sequence(ak4CaloJetSequence_mix)

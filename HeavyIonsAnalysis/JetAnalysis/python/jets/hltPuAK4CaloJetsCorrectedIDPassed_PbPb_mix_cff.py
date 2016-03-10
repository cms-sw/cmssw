import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

hltPuAK4CaloJetsCorrectedIDPassedmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassed"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

hltPuAK4CaloJetsCorrectedIDPassedparton = patJetPartonMatch.clone(src = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassed"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

hltPuAK4CaloJetsCorrectedIDPassedcorr = patJetCorrFactors.clone(
    useNPV = False,
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassed"),
    payload = "AK4CaloHLT"
    )

hltPuAK4CaloJetsCorrectedIDPassedpatJets = patJets.clone(jetSource = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassed"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("")),
                                               genJetMatch = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassedmatch"),
                                               genPartonMatch = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassedparton"),
                                               jetIDMap = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassedJetID"),
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
                              addJetCorrFactors = False
                              )


hltPuAK4CaloJetsCorrectedIDPassedJetAnalyzer = inclusiveJetAnalyzer.clone(
    jetTag = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassedpatJets"),
#    jetTag = cms.InputTag("hltPuAK4CaloJetsCorrectedIDPassed"),
                                               genjetTag = 'ak4HiGenJets',
                                               rParam = 0.4,
                                               matchJets = cms.untracked.bool(False),
                                               matchTag = 'patJets',
                                               pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                               trackTag = cms.InputTag("hiGeneralTracks"),
                                               fillGenJets = True,
                                               isMC = True,
                                               genParticles = cms.untracked.InputTag("genParticles"),
                                               eventInfoTag = cms.InputTag("generator"),
                                               doHiJetID = False,
    useJEC = cms.untracked.bool(False),
                                                             )


hltPuAK4CaloJetsCorrectedIDPassedJetSequence_mc = cms.Sequence(
                                                  hltPuAK4CaloJetsCorrectedIDPassedmatch
                                                  *
                                                  hltPuAK4CaloJetsCorrectedIDPassedparton
                                                  *
                                                  hltPuAK4CaloJetsCorrectedIDPassedcorr
                                                  *
                                                  hltPuAK4CaloJetsCorrectedIDPassedpatJets
                                                  *
                                                  hltPuAK4CaloJetsCorrectedIDPassedJetAnalyzer
                                                  )

hltPuAK4CaloJetsCorrectedIDPassedJetSequence_data = cms.Sequence(hltPuAK4CaloJetsCorrectedIDPassedcorr
                                                    *
                                                    hltPuAK4CaloJetsCorrectedIDPassedpatJets
                                                    *
                                                    hltPuAK4CaloJetsCorrectedIDPassedJetAnalyzer
                                                    )

hltPuAK4CaloJetsCorrectedIDPassedJetSequence_jec = cms.Sequence(hltPuAK4CaloJetsCorrectedIDPassedJetSequence_mc)
hltPuAK4CaloJetsCorrectedIDPassedJetSequence_mix = cms.Sequence(hltPuAK4CaloJetsCorrectedIDPassedJetSequence_mc)

hltPuAK4CaloJetsCorrectedIDPassedJetSequence = cms.Sequence(hltPuAK4CaloJetsCorrectedIDPassedJetSequence_mix)








                                          

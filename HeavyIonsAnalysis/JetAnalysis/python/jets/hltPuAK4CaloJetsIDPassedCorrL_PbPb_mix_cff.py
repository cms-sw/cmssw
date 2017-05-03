import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

hltPuAK4CaloJetsIDPassedCorrLmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

hltPuAK4CaloJetsIDPassedCorrLparton = patJetPartonMatch.clone(src = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

hltPuAK4CaloJetsIDPassedCorrLcorr = patJetCorrFactors.clone(
    useNPV = False,
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
    payload = "AK4Calo"
    )

hltPuAK4CaloJetsIDPassedCorrLpatJets = patJets.clone(jetSource = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("hltPuAK4CaloJetsIDPassedCorrLcorr")),
                                               genJetMatch = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrLmatch"),
                                               genPartonMatch = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrLparton"),
                                               jetIDMap = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrLJetID"),
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
                              )


hltPuAK4CaloJetsIDPassedCorrLJetAnalyzer = inclusiveJetAnalyzer.clone(
    jetTag = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrLpatJets"),
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
    )


hltPuAK4CaloJetsIDPassedCorrLJetSequence_mc = cms.Sequence(
                                                  hltPuAK4CaloJetsIDPassedCorrLmatch
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrLparton
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrLcorr
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrLpatJets
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrLJetAnalyzer
                                                  )

hltPuAK4CaloJetsIDPassedCorrLJetSequence_data = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrLcorr
                                                    *
                                                    hltPuAK4CaloJetsIDPassedCorrLpatJets
                                                    *
                                                    hltPuAK4CaloJetsIDPassedCorrLJetAnalyzer
                                                    )

hltPuAK4CaloJetsIDPassedCorrLJetSequence_jec = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrLJetSequence_mc)
hltPuAK4CaloJetsIDPassedCorrLJetSequence_mix = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrLJetSequence_mc)

hltPuAK4CaloJetsIDPassedCorrLJetSequence = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrLJetSequence_mix)








                                          

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *

hltPuAK4CaloJetsIDPassedCorrKmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

hltPuAK4CaloJetsIDPassedCorrKparton = patJetPartonMatch.clone(src = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
                                                        matched = cms.InputTag("genParticles")
                                                        )

hltPuAK4CaloJetsIDPassedCorrKcorr = patJetCorrFactors.clone(
    useNPV = False,
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
    payload = "AK4CaloHLT"
    )

hltPuAK4CaloJetsIDPassedCorrKpatJets = patJets.clone(jetSource = cms.InputTag("hltPuAK4CaloJetsIDPassed"),
                                               jetCorrFactorsSource = cms.VInputTag(cms.InputTag("hltPuAK4CaloJetsIDPassedCorrKcorr")),
                                               genJetMatch = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrKmatch"),
                                               genPartonMatch = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrKparton"),
                                               jetIDMap = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrKJetID"),
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


hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer = inclusiveJetAnalyzer.clone(
    jetTag = cms.InputTag("hltPuAK4CaloJetsIDPassedCorrKpatJets"),
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


hltPuAK4CaloJetsIDPassedCorrKJetSequence_mc = cms.Sequence(
                                                  hltPuAK4CaloJetsIDPassedCorrKmatch
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrKparton
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrKcorr
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrKpatJets
                                                  *
                                                  hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer
                                                  )

hltPuAK4CaloJetsIDPassedCorrKJetSequence_data = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrKcorr
                                                    *
                                                    hltPuAK4CaloJetsIDPassedCorrKpatJets
                                                    *
                                                    hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer
                                                    )

hltPuAK4CaloJetsIDPassedCorrKJetSequence_jec = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrKJetSequence_mc)
hltPuAK4CaloJetsIDPassedCorrKJetSequence_mix = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrKJetSequence_mc)

hltPuAK4CaloJetsIDPassedCorrKJetSequence = cms.Sequence(hltPuAK4CaloJetsIDPassedCorrKJetSequence_mix)








                                          

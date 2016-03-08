import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

ak4PFRawmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4PFJets"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

ak4PFRawparton = patJetPartonMatch.clone(src = cms.InputTag("ak4PFJets"))

ak4PFRawcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring(),#'L2Relative','L3Absolute'),
    src = cms.InputTag("ak4PFJets"),
    payload = "AK4PF_offline"
    )


ak4PFJetRawID = cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak4PFJets'))

#ak4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

ak4PFpatJetsRaw = patJets.clone(jetSource = cms.InputTag("ak4PFJets"),
        genJetMatch          = cms.InputTag("ak4PFRawmatch"),
        genPartonMatch       = cms.InputTag("ak4PFRawparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak4PFRawcorr")),                                       
        jetIDMap = cms.InputTag("ak4PFJetRawID"),
        addBTagInfo = False,
        addTagInfos = False,
        addDiscriminators = False,
        addAssociatedTracks = False,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = False,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

ak4PFRawJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak4PFpatJetsRaw"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsRaw',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             jetName = cms.untracked.string("ak4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
                                                             doSubJets = True

                                                             #gentau1 = cms.InputTag("ak4HiGenNjettiness","tau1"),
                                                             #gentau2 = cms.InputTag("ak4HiGenNjettiness","tau2"),
                                                             #gentau3 = cms.InputTag("ak4HiGenNjettiness","tau3")
                                                             )

ak4PFRawJetSequence_mc = cms.Sequence(
                                                  #ak4PFclean
                                                  #*
                                                  ak4PFRawmatch
                                                  *
                                                  ak4PFRawparton
                                                  *
                                                  ak4PFRawcorr
                                                  *
                                                  #ak4PFJetID
                                                  #*
                                                  #ak4PFRawPatJetFlavourIdLegacy
                                                  #*
                                                  ak4PFpatJetsRaw
 						  *
						  ak4PFRawJetAnalyzer
                                                  )

ak4PFRawJetSequence_data = cms.Sequence(ak4PFRawcorr
                                                    *
                                                    #ak4PFJetID
                                                    #*
                                                    ak4PFpatJetsRaw
                                                    *
						    ak4PFRawJetAnalyzer
                                                    )

ak4PFRawJetSequence_jec = cms.Sequence(ak4PFRawJetSequence_mc)
ak4PFRawJetSequence_mix = cms.Sequence(ak4PFRawJetSequence_mc)

ak4PFRawJetSequence = cms.Sequence(ak4PFRawJetSequence_mc)

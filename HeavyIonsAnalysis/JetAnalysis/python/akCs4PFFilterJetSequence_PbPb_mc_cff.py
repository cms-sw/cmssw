import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs4PFFiltermatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs4PFJetsFilter"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

akCs4PFFilterparton = patJetPartonMatch.clone(src = cms.InputTag("akCs4PFJetsFilter")
                                                        )

akCs4PFFiltercorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs4PFJetsFilter"),
    payload = "AK4PF_offline"
    )

akCs4PFJetFilterID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs4PFJetsFilter'))

#akCs4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

akCs4PFpatJetsFilter = patJets.clone(jetSource = cms.InputTag("akCs4PFJetsFilter"),
        genJetMatch          = cms.InputTag("akCs4PFFiltermatch"),
        genPartonMatch       = cms.InputTag("akCs4PFFilterparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs4PFFiltercorr")),                                       
        jetIDMap = cms.InputTag("akCs4PFJetFilterID"),
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

akCs4PFFilterJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs4PFpatJetsFilter"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsFilter',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             jetName = cms.untracked.string("akCs4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
                                                             doSubJets = True

                                                             #gentau1 = cms.InputTag("ak4HiGenNjettiness","tau1"),
                                                             #gentau2 = cms.InputTag("ak4HiGenNjettiness","tau2"),
                                                             #gentau3 = cms.InputTag("ak4HiGenNjettiness","tau3")
                                                             )

akCs4PFFilterJetSequence_mc = cms.Sequence(
                                                  #akCs4PFclean
                                                  #*
                                                  akCs4PFFiltermatch
                                                  *
                                                  akCs4PFFilterparton
                                                  *
                                                  akCs4PFFiltercorr
                                                  *
                                                  #akCs4PFJetID
                                                  #*
                                                  #akCs4PFFilterPatJetFlavourIdLegacy
                                                  #*
                                                  akCs4PFpatJetsFilter
 						  *
						  akCs4PFFilterJetAnalyzer
                                                  )

akCs4PFFilterJetSequence_data = cms.Sequence(akCs4PFFiltercorr
                                                    *
                                                    #akCs4PFJetID
                                                    #*
                                                    akCs4PFpatJetsFilter
                                                    *
						    akCs4PFFilterJetAnalyzer
                                                    )

akCs4PFFilterJetSequence_jec = cms.Sequence(akCs4PFFilterJetSequence_mc)
akCs4PFFilterJetSequence_mix = cms.Sequence(akCs4PFFilterJetSequence_mc)

akCs4PFFilterJetSequence = cms.Sequence(akCs4PFFilterJetSequence_mc)

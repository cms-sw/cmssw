import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs4PFSoftDropmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs4PFJetsSoftDrop"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

akCs4PFSoftDropparton = patJetPartonMatch.clone(src = cms.InputTag("akCs4PFJetsSoftDrop")
                                                        )

akCs4PFSoftDropcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs4PFJetsSoftDrop"),
    payload = "AK4PF_offline"
    )

akCs4PFJetSoftDropID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs4PFJetsSoftDrop'))

#akCs4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

akCs4PFpatJetsSoftDrop = patJets.clone(jetSource = cms.InputTag("akCs4PFJetsSoftDrop"),
        genJetMatch          = cms.InputTag("akCs4PFSoftDropmatch"),
        genPartonMatch       = cms.InputTag("akCs4PFSoftDropparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs4PFSoftDropcorr")),                                       
        jetIDMap = cms.InputTag("akCs4PFJetSoftDropID"),
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

akCs4PFSoftDropJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs4PFpatJetsSoftDrop"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsSoftDrop',
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

akCs4PFSoftDropJetSequence_mc = cms.Sequence(
                                                  #akCs4PFclean
                                                  #*
                                                  akCs4PFSoftDropmatch
                                                  *
                                                  akCs4PFSoftDropparton
                                                  *
                                                  akCs4PFSoftDropcorr
                                                  *
                                                  #akCs4PFJetID
                                                  #*
                                                  #akCs4PFSoftDropPatJetFlavourIdLegacy
                                                  #*
                                                  akCs4PFpatJetsSoftDrop
 						  *
						  akCs4PFSoftDropJetAnalyzer
                                                  )

akCs4PFSoftDropJetSequence_data = cms.Sequence(akCs4PFSoftDropcorr
                                                    *
                                                    #akCs4PFJetID
                                                    #*
                                                    akCs4PFpatJetsSoftDrop
                                                    *
						    akCs4PFSoftDropJetAnalyzer
                                                    )

akCs4PFSoftDropJetSequence_jec = cms.Sequence(akCs4PFSoftDropJetSequence_mc)
akCs4PFSoftDropJetSequence_mix = cms.Sequence(akCs4PFSoftDropJetSequence_mc)

akCs4PFSoftDropJetSequence = cms.Sequence(akCs4PFSoftDropJetSequence_mc)

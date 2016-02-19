

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akVs2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

akVs2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2CaloJets")
                                                        )

akVs2Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs2CaloJets"),
    payload = "AK2Calo_offline"
    )

akVs2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs2CaloJets'))

#akVs2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

akVs2CalobTagger = bTaggers("akVs2Calo",0.2)

#create objects locally since they dont load properly otherwise
#akVs2Calomatch = akVs2CalobTagger.match
akVs2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2CaloJets"), matched = cms.InputTag("genParticles"))
akVs2CaloPatJetFlavourAssociationLegacy = akVs2CalobTagger.PatJetFlavourAssociationLegacy
akVs2CaloPatJetPartons = akVs2CalobTagger.PatJetPartons
akVs2CaloJetTracksAssociatorAtVertex = akVs2CalobTagger.JetTracksAssociatorAtVertex
akVs2CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs2CaloSimpleSecondaryVertexHighEffBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2CaloSimpleSecondaryVertexHighPurBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2CaloCombinedSecondaryVertexBJetTags = akVs2CalobTagger.CombinedSecondaryVertexBJetTags
akVs2CaloCombinedSecondaryVertexV2BJetTags = akVs2CalobTagger.CombinedSecondaryVertexV2BJetTags
akVs2CaloJetBProbabilityBJetTags = akVs2CalobTagger.JetBProbabilityBJetTags
akVs2CaloSoftPFMuonByPtBJetTags = akVs2CalobTagger.SoftPFMuonByPtBJetTags
akVs2CaloSoftPFMuonByIP3dBJetTags = akVs2CalobTagger.SoftPFMuonByIP3dBJetTags
akVs2CaloTrackCountingHighEffBJetTags = akVs2CalobTagger.TrackCountingHighEffBJetTags
akVs2CaloTrackCountingHighPurBJetTags = akVs2CalobTagger.TrackCountingHighPurBJetTags
akVs2CaloPatJetPartonAssociationLegacy = akVs2CalobTagger.PatJetPartonAssociationLegacy

akVs2CaloImpactParameterTagInfos = akVs2CalobTagger.ImpactParameterTagInfos
akVs2CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs2CaloJetProbabilityBJetTags = akVs2CalobTagger.JetProbabilityBJetTags
akVs2CaloPositiveOnlyJetProbabilityBJetTags = akVs2CalobTagger.PositiveOnlyJetProbabilityBJetTags
akVs2CaloNegativeOnlyJetProbabilityBJetTags = akVs2CalobTagger.NegativeOnlyJetProbabilityBJetTags
akVs2CaloNegativeTrackCountingHighEffBJetTags = akVs2CalobTagger.NegativeTrackCountingHighEffBJetTags
akVs2CaloNegativeTrackCountingHighPurBJetTags = akVs2CalobTagger.NegativeTrackCountingHighPurBJetTags
akVs2CaloNegativeOnlyJetBProbabilityBJetTags = akVs2CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akVs2CaloPositiveOnlyJetBProbabilityBJetTags = akVs2CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akVs2CaloSecondaryVertexTagInfos = akVs2CalobTagger.SecondaryVertexTagInfos
akVs2CaloSimpleSecondaryVertexHighEffBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2CaloSimpleSecondaryVertexHighPurBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2CaloCombinedSecondaryVertexBJetTags = akVs2CalobTagger.CombinedSecondaryVertexBJetTags
akVs2CaloCombinedSecondaryVertexV2BJetTags = akVs2CalobTagger.CombinedSecondaryVertexV2BJetTags

akVs2CaloSecondaryVertexNegativeTagInfos = akVs2CalobTagger.SecondaryVertexNegativeTagInfos
akVs2CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akVs2CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs2CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akVs2CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs2CaloNegativeCombinedSecondaryVertexBJetTags = akVs2CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akVs2CaloPositiveCombinedSecondaryVertexBJetTags = akVs2CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akVs2CaloSoftPFMuonsTagInfos = akVs2CalobTagger.SoftPFMuonsTagInfos
akVs2CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs2CaloSoftPFMuonBJetTags = akVs2CalobTagger.SoftPFMuonBJetTags
akVs2CaloSoftPFMuonByIP3dBJetTags = akVs2CalobTagger.SoftPFMuonByIP3dBJetTags
akVs2CaloSoftPFMuonByPtBJetTags = akVs2CalobTagger.SoftPFMuonByPtBJetTags
akVs2CaloNegativeSoftPFMuonByPtBJetTags = akVs2CalobTagger.NegativeSoftPFMuonByPtBJetTags
akVs2CaloPositiveSoftPFMuonByPtBJetTags = akVs2CalobTagger.PositiveSoftPFMuonByPtBJetTags
akVs2CaloPatJetFlavourIdLegacy = cms.Sequence(akVs2CaloPatJetPartonAssociationLegacy*akVs2CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs2CaloPatJetFlavourAssociation = akVs2CalobTagger.PatJetFlavourAssociation
#akVs2CaloPatJetFlavourId = cms.Sequence(akVs2CaloPatJetPartons*akVs2CaloPatJetFlavourAssociation)

akVs2CaloJetBtaggingIP       = cms.Sequence(akVs2CaloImpactParameterTagInfos *
            (akVs2CaloTrackCountingHighEffBJetTags +
             akVs2CaloTrackCountingHighPurBJetTags +
             akVs2CaloJetProbabilityBJetTags +
             akVs2CaloJetBProbabilityBJetTags +
             akVs2CaloPositiveOnlyJetProbabilityBJetTags +
             akVs2CaloNegativeOnlyJetProbabilityBJetTags +
             akVs2CaloNegativeTrackCountingHighEffBJetTags +
             akVs2CaloNegativeTrackCountingHighPurBJetTags +
             akVs2CaloNegativeOnlyJetBProbabilityBJetTags +
             akVs2CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs2CaloJetBtaggingSV = cms.Sequence(akVs2CaloImpactParameterTagInfos
            *
            akVs2CaloSecondaryVertexTagInfos
            * (akVs2CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs2CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs2CaloCombinedSecondaryVertexBJetTags
                +
                akVs2CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akVs2CaloJetBtaggingNegSV = cms.Sequence(akVs2CaloImpactParameterTagInfos
            *
            akVs2CaloSecondaryVertexNegativeTagInfos
            * (akVs2CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs2CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs2CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akVs2CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs2CaloJetBtaggingMu = cms.Sequence(akVs2CaloSoftPFMuonsTagInfos * (akVs2CaloSoftPFMuonBJetTags
                +
                akVs2CaloSoftPFMuonByIP3dBJetTags
                +
                akVs2CaloSoftPFMuonByPtBJetTags
                +
                akVs2CaloNegativeSoftPFMuonByPtBJetTags
                +
                akVs2CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs2CaloJetBtagging = cms.Sequence(akVs2CaloJetBtaggingIP
            *akVs2CaloJetBtaggingSV
            *akVs2CaloJetBtaggingNegSV
#            *akVs2CaloJetBtaggingMu
            )

akVs2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs2CaloJets"),
        genJetMatch          = cms.InputTag("akVs2Calomatch"),
        genPartonMatch       = cms.InputTag("akVs2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs2CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs2CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs2CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs2CaloJetProbabilityBJetTags"),
            #cms.InputTag("akVs2CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs2CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs2CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akVs2CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akVs2CaloJets"),
           	    R0  = cms.double( 0.2)
)
akVs2CalopatJetsWithBtagging.userData.userFloats.src += ['akVs2CaloNjettiness:tau1','akVs2CaloNjettiness:tau2','akVs2CaloNjettiness:tau3']

akVs2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akVs2Calo"),
                                                             jetName = cms.untracked.string("akVs2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akVs2CaloJetSequence_mc = cms.Sequence(
                                                  #akVs2Caloclean
                                                  #*
                                                  akVs2Calomatch
                                                  *
                                                  akVs2Caloparton
                                                  *
                                                  akVs2Calocorr
                                                  *
                                                  #akVs2CaloJetID
                                                  #*
                                                  akVs2CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akVs2CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs2CaloJetBtagging
                                                  *
                                                  akVs2CaloNjettiness
                                                  *
                                                  akVs2CalopatJetsWithBtagging
                                                  *
                                                  akVs2CaloJetAnalyzer
                                                  )

akVs2CaloJetSequence_data = cms.Sequence(akVs2Calocorr
                                                    *
                                                    #akVs2CaloJetID
                                                    #*
                                                    akVs2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs2CaloJetBtagging
                                                    *
                                                    akVs2CaloNjettiness 
                                                    *
                                                    akVs2CalopatJetsWithBtagging
                                                    *
                                                    akVs2CaloJetAnalyzer
                                                    )

akVs2CaloJetSequence_jec = cms.Sequence(akVs2CaloJetSequence_mc)
akVs2CaloJetSequence_mix = cms.Sequence(akVs2CaloJetSequence_mc)

akVs2CaloJetSequence = cms.Sequence(akVs2CaloJetSequence_mc)



import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs2CaloJets"),
    matched = cms.InputTag("ak2GenJets"),
    maxDeltaR = 0.2
    )

akCs2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs2CaloJets")
                                                        )

akCs2Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs2CaloJets"),
    payload = "AK2Calo_offline"
    )

akCs2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs2CaloJets'))

#akCs2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2GenJets'))

akCs2CalobTagger = bTaggers("akCs2Calo",0.2)

#create objects locally since they dont load properly otherwise
#akCs2Calomatch = akCs2CalobTagger.match
akCs2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs2CaloJets"), matched = cms.InputTag("selectedPartons"))
akCs2CaloPatJetFlavourAssociationLegacy = akCs2CalobTagger.PatJetFlavourAssociationLegacy
akCs2CaloPatJetPartons = akCs2CalobTagger.PatJetPartons
akCs2CaloJetTracksAssociatorAtVertex = akCs2CalobTagger.JetTracksAssociatorAtVertex
akCs2CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs2CaloSimpleSecondaryVertexHighEffBJetTags = akCs2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs2CaloSimpleSecondaryVertexHighPurBJetTags = akCs2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs2CaloCombinedSecondaryVertexBJetTags = akCs2CalobTagger.CombinedSecondaryVertexBJetTags
akCs2CaloCombinedSecondaryVertexV2BJetTags = akCs2CalobTagger.CombinedSecondaryVertexV2BJetTags
akCs2CaloJetBProbabilityBJetTags = akCs2CalobTagger.JetBProbabilityBJetTags
akCs2CaloSoftPFMuonByPtBJetTags = akCs2CalobTagger.SoftPFMuonByPtBJetTags
akCs2CaloSoftPFMuonByIP3dBJetTags = akCs2CalobTagger.SoftPFMuonByIP3dBJetTags
akCs2CaloTrackCountingHighEffBJetTags = akCs2CalobTagger.TrackCountingHighEffBJetTags
akCs2CaloTrackCountingHighPurBJetTags = akCs2CalobTagger.TrackCountingHighPurBJetTags
akCs2CaloPatJetPartonAssociationLegacy = akCs2CalobTagger.PatJetPartonAssociationLegacy

akCs2CaloImpactParameterTagInfos = akCs2CalobTagger.ImpactParameterTagInfos
akCs2CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs2CaloJetProbabilityBJetTags = akCs2CalobTagger.JetProbabilityBJetTags
akCs2CaloPositiveOnlyJetProbabilityBJetTags = akCs2CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCs2CaloNegativeOnlyJetProbabilityBJetTags = akCs2CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCs2CaloNegativeTrackCountingHighEffBJetTags = akCs2CalobTagger.NegativeTrackCountingHighEffBJetTags
akCs2CaloNegativeTrackCountingHighPurBJetTags = akCs2CalobTagger.NegativeTrackCountingHighPurBJetTags
akCs2CaloNegativeOnlyJetBProbabilityBJetTags = akCs2CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCs2CaloPositiveOnlyJetBProbabilityBJetTags = akCs2CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCs2CaloSecondaryVertexTagInfos = akCs2CalobTagger.SecondaryVertexTagInfos
akCs2CaloSimpleSecondaryVertexHighEffBJetTags = akCs2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs2CaloSimpleSecondaryVertexHighPurBJetTags = akCs2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs2CaloCombinedSecondaryVertexBJetTags = akCs2CalobTagger.CombinedSecondaryVertexBJetTags
akCs2CaloCombinedSecondaryVertexV2BJetTags = akCs2CalobTagger.CombinedSecondaryVertexV2BJetTags

akCs2CaloSecondaryVertexNegativeTagInfos = akCs2CalobTagger.SecondaryVertexNegativeTagInfos
akCs2CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCs2CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs2CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCs2CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs2CaloNegativeCombinedSecondaryVertexBJetTags = akCs2CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCs2CaloPositiveCombinedSecondaryVertexBJetTags = akCs2CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCs2CaloSoftPFMuonsTagInfos = akCs2CalobTagger.SoftPFMuonsTagInfos
akCs2CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs2CaloSoftPFMuonBJetTags = akCs2CalobTagger.SoftPFMuonBJetTags
akCs2CaloSoftPFMuonByIP3dBJetTags = akCs2CalobTagger.SoftPFMuonByIP3dBJetTags
akCs2CaloSoftPFMuonByPtBJetTags = akCs2CalobTagger.SoftPFMuonByPtBJetTags
akCs2CaloNegativeSoftPFMuonByPtBJetTags = akCs2CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCs2CaloPositiveSoftPFMuonByPtBJetTags = akCs2CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCs2CaloPatJetFlavourIdLegacy = cms.Sequence(akCs2CaloPatJetPartonAssociationLegacy*akCs2CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs2CaloPatJetFlavourAssociation = akCs2CalobTagger.PatJetFlavourAssociation
#akCs2CaloPatJetFlavourId = cms.Sequence(akCs2CaloPatJetPartons*akCs2CaloPatJetFlavourAssociation)

akCs2CaloJetBtaggingIP       = cms.Sequence(akCs2CaloImpactParameterTagInfos *
            (akCs2CaloTrackCountingHighEffBJetTags +
             akCs2CaloTrackCountingHighPurBJetTags +
             akCs2CaloJetProbabilityBJetTags +
             akCs2CaloJetBProbabilityBJetTags +
             akCs2CaloPositiveOnlyJetProbabilityBJetTags +
             akCs2CaloNegativeOnlyJetProbabilityBJetTags +
             akCs2CaloNegativeTrackCountingHighEffBJetTags +
             akCs2CaloNegativeTrackCountingHighPurBJetTags +
             akCs2CaloNegativeOnlyJetBProbabilityBJetTags +
             akCs2CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCs2CaloJetBtaggingSV = cms.Sequence(akCs2CaloImpactParameterTagInfos
            *
            akCs2CaloSecondaryVertexTagInfos
            * (akCs2CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCs2CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCs2CaloCombinedSecondaryVertexBJetTags
                +
                akCs2CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCs2CaloJetBtaggingNegSV = cms.Sequence(akCs2CaloImpactParameterTagInfos
            *
            akCs2CaloSecondaryVertexNegativeTagInfos
            * (akCs2CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCs2CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCs2CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCs2CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCs2CaloJetBtaggingMu = cms.Sequence(akCs2CaloSoftPFMuonsTagInfos * (akCs2CaloSoftPFMuonBJetTags
                +
                akCs2CaloSoftPFMuonByIP3dBJetTags
                +
                akCs2CaloSoftPFMuonByPtBJetTags
                +
                akCs2CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCs2CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs2CaloJetBtagging = cms.Sequence(akCs2CaloJetBtaggingIP
            *akCs2CaloJetBtaggingSV
            *akCs2CaloJetBtaggingNegSV
#            *akCs2CaloJetBtaggingMu
            )

akCs2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs2CaloJets"),
        genJetMatch          = cms.InputTag("akCs2Calomatch"),
        genPartonMatch       = cms.InputTag("akCs2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCs2CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs2CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs2CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCs2CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCs2CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs2CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs2CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akCs2CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs2CaloJets"),
           	    R0  = cms.double( 0.2)
)
akCs2CalopatJetsWithBtagging.userData.userFloats.src += ['akCs2CaloNjettiness:tau1','akCs2CaloNjettiness:tau2','akCs2CaloNjettiness:tau3']

akCs2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2GenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCs2Calo"),
                                                             jetName = cms.untracked.string("akCs2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs2CaloJetSequence_mc = cms.Sequence(
                                                  #akCs2Caloclean
                                                  #*
                                                  akCs2Calomatch
                                                  *
                                                  akCs2Caloparton
                                                  *
                                                  akCs2Calocorr
                                                  *
                                                  #akCs2CaloJetID
                                                  #*
                                                  akCs2CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCs2CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCs2CaloJetBtagging
                                                  *
                                                  akCs2CaloNjettiness
                                                  *
                                                  akCs2CalopatJetsWithBtagging
                                                  *
                                                  akCs2CaloJetAnalyzer
                                                  )

akCs2CaloJetSequence_data = cms.Sequence(akCs2Calocorr
                                                    *
                                                    #akCs2CaloJetID
                                                    #*
                                                    akCs2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCs2CaloJetBtagging
                                                    *
                                                    akCs2CaloNjettiness 
                                                    *
                                                    akCs2CalopatJetsWithBtagging
                                                    *
                                                    akCs2CaloJetAnalyzer
                                                    )

akCs2CaloJetSequence_jec = cms.Sequence(akCs2CaloJetSequence_mc)
akCs2CaloJetSequence_mb = cms.Sequence(akCs2CaloJetSequence_mc)

akCs2CaloJetSequence = cms.Sequence(akCs2CaloJetSequence_data)

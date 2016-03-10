

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs6CaloJets"),
    matched = cms.InputTag("ak6GenJets"),
    maxDeltaR = 0.6
    )

akCs6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs6CaloJets")
                                                        )

akCs6Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs6CaloJets"),
    payload = "AK6Calo_offline"
    )

akCs6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs6CaloJets'))

#akCs6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6GenJets'))

akCs6CalobTagger = bTaggers("akCs6Calo",0.6)

#create objects locally since they dont load properly otherwise
#akCs6Calomatch = akCs6CalobTagger.match
akCs6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs6CaloJets"), matched = cms.InputTag("selectedPartons"))
akCs6CaloPatJetFlavourAssociationLegacy = akCs6CalobTagger.PatJetFlavourAssociationLegacy
akCs6CaloPatJetPartons = akCs6CalobTagger.PatJetPartons
akCs6CaloJetTracksAssociatorAtVertex = akCs6CalobTagger.JetTracksAssociatorAtVertex
akCs6CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs6CaloSimpleSecondaryVertexHighEffBJetTags = akCs6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs6CaloSimpleSecondaryVertexHighPurBJetTags = akCs6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs6CaloCombinedSecondaryVertexBJetTags = akCs6CalobTagger.CombinedSecondaryVertexBJetTags
akCs6CaloCombinedSecondaryVertexV2BJetTags = akCs6CalobTagger.CombinedSecondaryVertexV2BJetTags
akCs6CaloJetBProbabilityBJetTags = akCs6CalobTagger.JetBProbabilityBJetTags
akCs6CaloSoftPFMuonByPtBJetTags = akCs6CalobTagger.SoftPFMuonByPtBJetTags
akCs6CaloSoftPFMuonByIP3dBJetTags = akCs6CalobTagger.SoftPFMuonByIP3dBJetTags
akCs6CaloTrackCountingHighEffBJetTags = akCs6CalobTagger.TrackCountingHighEffBJetTags
akCs6CaloTrackCountingHighPurBJetTags = akCs6CalobTagger.TrackCountingHighPurBJetTags
akCs6CaloPatJetPartonAssociationLegacy = akCs6CalobTagger.PatJetPartonAssociationLegacy

akCs6CaloImpactParameterTagInfos = akCs6CalobTagger.ImpactParameterTagInfos
akCs6CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs6CaloJetProbabilityBJetTags = akCs6CalobTagger.JetProbabilityBJetTags
akCs6CaloPositiveOnlyJetProbabilityBJetTags = akCs6CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCs6CaloNegativeOnlyJetProbabilityBJetTags = akCs6CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCs6CaloNegativeTrackCountingHighEffBJetTags = akCs6CalobTagger.NegativeTrackCountingHighEffBJetTags
akCs6CaloNegativeTrackCountingHighPurBJetTags = akCs6CalobTagger.NegativeTrackCountingHighPurBJetTags
akCs6CaloNegativeOnlyJetBProbabilityBJetTags = akCs6CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCs6CaloPositiveOnlyJetBProbabilityBJetTags = akCs6CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCs6CaloSecondaryVertexTagInfos = akCs6CalobTagger.SecondaryVertexTagInfos
akCs6CaloSimpleSecondaryVertexHighEffBJetTags = akCs6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs6CaloSimpleSecondaryVertexHighPurBJetTags = akCs6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs6CaloCombinedSecondaryVertexBJetTags = akCs6CalobTagger.CombinedSecondaryVertexBJetTags
akCs6CaloCombinedSecondaryVertexV2BJetTags = akCs6CalobTagger.CombinedSecondaryVertexV2BJetTags

akCs6CaloSecondaryVertexNegativeTagInfos = akCs6CalobTagger.SecondaryVertexNegativeTagInfos
akCs6CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCs6CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs6CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCs6CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs6CaloNegativeCombinedSecondaryVertexBJetTags = akCs6CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCs6CaloPositiveCombinedSecondaryVertexBJetTags = akCs6CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCs6CaloSoftPFMuonsTagInfos = akCs6CalobTagger.SoftPFMuonsTagInfos
akCs6CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs6CaloSoftPFMuonBJetTags = akCs6CalobTagger.SoftPFMuonBJetTags
akCs6CaloSoftPFMuonByIP3dBJetTags = akCs6CalobTagger.SoftPFMuonByIP3dBJetTags
akCs6CaloSoftPFMuonByPtBJetTags = akCs6CalobTagger.SoftPFMuonByPtBJetTags
akCs6CaloNegativeSoftPFMuonByPtBJetTags = akCs6CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCs6CaloPositiveSoftPFMuonByPtBJetTags = akCs6CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCs6CaloPatJetFlavourIdLegacy = cms.Sequence(akCs6CaloPatJetPartonAssociationLegacy*akCs6CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs6CaloPatJetFlavourAssociation = akCs6CalobTagger.PatJetFlavourAssociation
#akCs6CaloPatJetFlavourId = cms.Sequence(akCs6CaloPatJetPartons*akCs6CaloPatJetFlavourAssociation)

akCs6CaloJetBtaggingIP       = cms.Sequence(akCs6CaloImpactParameterTagInfos *
            (akCs6CaloTrackCountingHighEffBJetTags +
             akCs6CaloTrackCountingHighPurBJetTags +
             akCs6CaloJetProbabilityBJetTags +
             akCs6CaloJetBProbabilityBJetTags +
             akCs6CaloPositiveOnlyJetProbabilityBJetTags +
             akCs6CaloNegativeOnlyJetProbabilityBJetTags +
             akCs6CaloNegativeTrackCountingHighEffBJetTags +
             akCs6CaloNegativeTrackCountingHighPurBJetTags +
             akCs6CaloNegativeOnlyJetBProbabilityBJetTags +
             akCs6CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCs6CaloJetBtaggingSV = cms.Sequence(akCs6CaloImpactParameterTagInfos
            *
            akCs6CaloSecondaryVertexTagInfos
            * (akCs6CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCs6CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCs6CaloCombinedSecondaryVertexBJetTags
                +
                akCs6CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCs6CaloJetBtaggingNegSV = cms.Sequence(akCs6CaloImpactParameterTagInfos
            *
            akCs6CaloSecondaryVertexNegativeTagInfos
            * (akCs6CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCs6CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCs6CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCs6CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCs6CaloJetBtaggingMu = cms.Sequence(akCs6CaloSoftPFMuonsTagInfos * (akCs6CaloSoftPFMuonBJetTags
                +
                akCs6CaloSoftPFMuonByIP3dBJetTags
                +
                akCs6CaloSoftPFMuonByPtBJetTags
                +
                akCs6CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCs6CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs6CaloJetBtagging = cms.Sequence(akCs6CaloJetBtaggingIP
            *akCs6CaloJetBtaggingSV
            *akCs6CaloJetBtaggingNegSV
#            *akCs6CaloJetBtaggingMu
            )

akCs6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs6CaloJets"),
        genJetMatch          = cms.InputTag("akCs6Calomatch"),
        genPartonMatch       = cms.InputTag("akCs6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs6Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCs6CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs6CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs6CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs6CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCs6CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCs6CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs6CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs6CaloJetID"),
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

akCs6CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs6CaloJets"),
           	    R0  = cms.double( 0.6)
)
akCs6CalopatJetsWithBtagging.userData.userFloats.src += ['akCs6CaloNjettiness:tau1','akCs6CaloNjettiness:tau2','akCs6CaloNjettiness:tau3']

akCs6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6GenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCs6Calo"),
                                                             jetName = cms.untracked.string("akCs6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs6CaloJetSequence_mc = cms.Sequence(
                                                  #akCs6Caloclean
                                                  #*
                                                  akCs6Calomatch
                                                  *
                                                  akCs6Caloparton
                                                  *
                                                  akCs6Calocorr
                                                  *
                                                  #akCs6CaloJetID
                                                  #*
                                                  akCs6CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCs6CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCs6CaloJetBtagging
                                                  *
                                                  akCs6CaloNjettiness
                                                  *
                                                  akCs6CalopatJetsWithBtagging
                                                  *
                                                  akCs6CaloJetAnalyzer
                                                  )

akCs6CaloJetSequence_data = cms.Sequence(akCs6Calocorr
                                                    *
                                                    #akCs6CaloJetID
                                                    #*
                                                    akCs6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCs6CaloJetBtagging
                                                    *
                                                    akCs6CaloNjettiness 
                                                    *
                                                    akCs6CalopatJetsWithBtagging
                                                    *
                                                    akCs6CaloJetAnalyzer
                                                    )

akCs6CaloJetSequence_jec = cms.Sequence(akCs6CaloJetSequence_mc)
akCs6CaloJetSequence_mb = cms.Sequence(akCs6CaloJetSequence_mc)

akCs6CaloJetSequence = cms.Sequence(akCs6CaloJetSequence_mc)

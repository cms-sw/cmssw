

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop4CaloJets"),
    matched = cms.InputTag("ak4HiSignalGenJets"),
    maxDeltaR = 0.4
    )

akCsSoftDrop4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop4CaloJets")
                                                        )

akCsSoftDrop4Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop4CaloJets"),
    payload = "AK4Calo_offline"
    )

akCsSoftDrop4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop4CaloJets'))

#akCsSoftDrop4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiSignalGenJets'))

akCsSoftDrop4CalobTagger = bTaggers("akCsSoftDrop4Calo",0.4)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop4Calomatch = akCsSoftDrop4CalobTagger.match
akCsSoftDrop4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop4CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop4CaloPatJetFlavourAssociationLegacy = akCsSoftDrop4CalobTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop4CaloPatJetPartons = akCsSoftDrop4CalobTagger.PatJetPartons
akCsSoftDrop4CaloJetTracksAssociatorAtVertex = akCsSoftDrop4CalobTagger.JetTracksAssociatorAtVertex
akCsSoftDrop4CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop4CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop4CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop4CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop4CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop4CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop4CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop4CaloJetBProbabilityBJetTags = akCsSoftDrop4CalobTagger.JetBProbabilityBJetTags
akCsSoftDrop4CaloSoftPFMuonByPtBJetTags = akCsSoftDrop4CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop4CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop4CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop4CaloTrackCountingHighEffBJetTags = akCsSoftDrop4CalobTagger.TrackCountingHighEffBJetTags
akCsSoftDrop4CaloTrackCountingHighPurBJetTags = akCsSoftDrop4CalobTagger.TrackCountingHighPurBJetTags
akCsSoftDrop4CaloPatJetPartonAssociationLegacy = akCsSoftDrop4CalobTagger.PatJetPartonAssociationLegacy

akCsSoftDrop4CaloImpactParameterTagInfos = akCsSoftDrop4CalobTagger.ImpactParameterTagInfos
akCsSoftDrop4CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop4CaloJetProbabilityBJetTags = akCsSoftDrop4CalobTagger.JetProbabilityBJetTags
akCsSoftDrop4CaloPositiveOnlyJetProbabilityBJetTags = akCsSoftDrop4CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCsSoftDrop4CaloNegativeOnlyJetProbabilityBJetTags = akCsSoftDrop4CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCsSoftDrop4CaloNegativeTrackCountingHighEffBJetTags = akCsSoftDrop4CalobTagger.NegativeTrackCountingHighEffBJetTags
akCsSoftDrop4CaloNegativeTrackCountingHighPurBJetTags = akCsSoftDrop4CalobTagger.NegativeTrackCountingHighPurBJetTags
akCsSoftDrop4CaloNegativeOnlyJetBProbabilityBJetTags = akCsSoftDrop4CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCsSoftDrop4CaloPositiveOnlyJetBProbabilityBJetTags = akCsSoftDrop4CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCsSoftDrop4CaloSecondaryVertexTagInfos = akCsSoftDrop4CalobTagger.SecondaryVertexTagInfos
akCsSoftDrop4CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop4CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop4CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop4CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop4CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop4CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop4CaloSecondaryVertexNegativeTagInfos = akCsSoftDrop4CalobTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop4CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop4CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop4CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop4CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop4CaloNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop4CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop4CaloPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop4CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCsSoftDrop4CaloSoftPFMuonsTagInfos = akCsSoftDrop4CalobTagger.SoftPFMuonsTagInfos
akCsSoftDrop4CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop4CaloSoftPFMuonBJetTags = akCsSoftDrop4CalobTagger.SoftPFMuonBJetTags
akCsSoftDrop4CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop4CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop4CaloSoftPFMuonByPtBJetTags = akCsSoftDrop4CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop4CaloNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop4CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop4CaloPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop4CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop4CaloPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop4CaloPatJetPartonAssociationLegacy*akCsSoftDrop4CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop4CaloPatJetFlavourAssociation = akCsSoftDrop4CalobTagger.PatJetFlavourAssociation
#akCsSoftDrop4CaloPatJetFlavourId = cms.Sequence(akCsSoftDrop4CaloPatJetPartons*akCsSoftDrop4CaloPatJetFlavourAssociation)

akCsSoftDrop4CaloJetBtaggingIP       = cms.Sequence(akCsSoftDrop4CaloImpactParameterTagInfos *
            (akCsSoftDrop4CaloTrackCountingHighEffBJetTags +
             akCsSoftDrop4CaloTrackCountingHighPurBJetTags +
             akCsSoftDrop4CaloJetProbabilityBJetTags +
             akCsSoftDrop4CaloJetBProbabilityBJetTags +
             akCsSoftDrop4CaloPositiveOnlyJetProbabilityBJetTags +
             akCsSoftDrop4CaloNegativeOnlyJetProbabilityBJetTags +
             akCsSoftDrop4CaloNegativeTrackCountingHighEffBJetTags +
             akCsSoftDrop4CaloNegativeTrackCountingHighPurBJetTags +
             akCsSoftDrop4CaloNegativeOnlyJetBProbabilityBJetTags +
             akCsSoftDrop4CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCsSoftDrop4CaloJetBtaggingSV = cms.Sequence(akCsSoftDrop4CaloImpactParameterTagInfos
            *
            akCsSoftDrop4CaloSecondaryVertexTagInfos
            * (akCsSoftDrop4CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCsSoftDrop4CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCsSoftDrop4CaloCombinedSecondaryVertexBJetTags
                +
                akCsSoftDrop4CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop4CaloJetBtaggingNegSV = cms.Sequence(akCsSoftDrop4CaloImpactParameterTagInfos
            *
            akCsSoftDrop4CaloSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop4CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCsSoftDrop4CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCsSoftDrop4CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCsSoftDrop4CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCsSoftDrop4CaloJetBtaggingMu = cms.Sequence(akCsSoftDrop4CaloSoftPFMuonsTagInfos * (akCsSoftDrop4CaloSoftPFMuonBJetTags
                +
                akCsSoftDrop4CaloSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop4CaloSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop4CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop4CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop4CaloJetBtagging = cms.Sequence(akCsSoftDrop4CaloJetBtaggingIP
            *akCsSoftDrop4CaloJetBtaggingSV
            *akCsSoftDrop4CaloJetBtaggingNegSV
#            *akCsSoftDrop4CaloJetBtaggingMu
            )

akCsSoftDrop4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop4CaloJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop4Calomatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop4Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop4CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop4CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop4CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop4CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop4CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop4CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop4CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop4CaloJetID"),
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

akCsSoftDrop4CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop4CaloJets"),
           	    R0  = cms.double( 0.4)
)
akCsSoftDrop4CalopatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop4CaloNjettiness:tau1','akCsSoftDrop4CaloNjettiness:tau2','akCsSoftDrop4CaloNjettiness:tau3']

akCsSoftDrop4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop4Calo"),
                                                             jetName = cms.untracked.string("akCsSoftDrop4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop4CaloJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop4Caloclean
                                                  #*
                                                  akCsSoftDrop4Calomatch
                                                  *
                                                  akCsSoftDrop4Caloparton
                                                  *
                                                  akCsSoftDrop4Calocorr
                                                  *
                                                  #akCsSoftDrop4CaloJetID
                                                  #*
                                                  akCsSoftDrop4CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop4CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop4CaloJetBtagging
                                                  *
                                                  akCsSoftDrop4CaloNjettiness
                                                  *
                                                  akCsSoftDrop4CalopatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop4CaloJetAnalyzer
                                                  )

akCsSoftDrop4CaloJetSequence_data = cms.Sequence(akCsSoftDrop4Calocorr
                                                    *
                                                    #akCsSoftDrop4CaloJetID
                                                    #*
                                                    akCsSoftDrop4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop4CaloJetBtagging
                                                    *
                                                    akCsSoftDrop4CaloNjettiness 
                                                    *
                                                    akCsSoftDrop4CalopatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop4CaloJetAnalyzer
                                                    )

akCsSoftDrop4CaloJetSequence_jec = cms.Sequence(akCsSoftDrop4CaloJetSequence_mc)
akCsSoftDrop4CaloJetSequence_mb = cms.Sequence(akCsSoftDrop4CaloJetSequence_mc)

akCsSoftDrop4CaloJetSequence = cms.Sequence(akCsSoftDrop4CaloJetSequence_data)

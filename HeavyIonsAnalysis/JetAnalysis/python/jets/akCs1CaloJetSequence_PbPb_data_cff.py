

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs1CaloJets"),
    matched = cms.InputTag("ak1HiSignalGenJets"),
    maxDeltaR = 0.1
    )

akCs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs1CaloJets")
                                                        )

akCs1Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs1CaloJets"),
    payload = "AK1Calo_offline"
    )

akCs1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs1CaloJets'))

#akCs1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiSignalGenJets'))

akCs1CalobTagger = bTaggers("akCs1Calo",0.1)

#create objects locally since they dont load properly otherwise
#akCs1Calomatch = akCs1CalobTagger.match
akCs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs1CaloJets"), matched = cms.InputTag("selectedPartons"))
akCs1CaloPatJetFlavourAssociationLegacy = akCs1CalobTagger.PatJetFlavourAssociationLegacy
akCs1CaloPatJetPartons = akCs1CalobTagger.PatJetPartons
akCs1CaloJetTracksAssociatorAtVertex = akCs1CalobTagger.JetTracksAssociatorAtVertex
akCs1CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs1CaloSimpleSecondaryVertexHighEffBJetTags = akCs1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs1CaloSimpleSecondaryVertexHighPurBJetTags = akCs1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs1CaloCombinedSecondaryVertexBJetTags = akCs1CalobTagger.CombinedSecondaryVertexBJetTags
akCs1CaloCombinedSecondaryVertexV2BJetTags = akCs1CalobTagger.CombinedSecondaryVertexV2BJetTags
akCs1CaloJetBProbabilityBJetTags = akCs1CalobTagger.JetBProbabilityBJetTags
akCs1CaloSoftPFMuonByPtBJetTags = akCs1CalobTagger.SoftPFMuonByPtBJetTags
akCs1CaloSoftPFMuonByIP3dBJetTags = akCs1CalobTagger.SoftPFMuonByIP3dBJetTags
akCs1CaloTrackCountingHighEffBJetTags = akCs1CalobTagger.TrackCountingHighEffBJetTags
akCs1CaloTrackCountingHighPurBJetTags = akCs1CalobTagger.TrackCountingHighPurBJetTags
akCs1CaloPatJetPartonAssociationLegacy = akCs1CalobTagger.PatJetPartonAssociationLegacy

akCs1CaloImpactParameterTagInfos = akCs1CalobTagger.ImpactParameterTagInfos
akCs1CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs1CaloJetProbabilityBJetTags = akCs1CalobTagger.JetProbabilityBJetTags
akCs1CaloPositiveOnlyJetProbabilityBJetTags = akCs1CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCs1CaloNegativeOnlyJetProbabilityBJetTags = akCs1CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCs1CaloNegativeTrackCountingHighEffBJetTags = akCs1CalobTagger.NegativeTrackCountingHighEffBJetTags
akCs1CaloNegativeTrackCountingHighPurBJetTags = akCs1CalobTagger.NegativeTrackCountingHighPurBJetTags
akCs1CaloNegativeOnlyJetBProbabilityBJetTags = akCs1CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCs1CaloPositiveOnlyJetBProbabilityBJetTags = akCs1CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCs1CaloSecondaryVertexTagInfos = akCs1CalobTagger.SecondaryVertexTagInfos
akCs1CaloSimpleSecondaryVertexHighEffBJetTags = akCs1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs1CaloSimpleSecondaryVertexHighPurBJetTags = akCs1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs1CaloCombinedSecondaryVertexBJetTags = akCs1CalobTagger.CombinedSecondaryVertexBJetTags
akCs1CaloCombinedSecondaryVertexV2BJetTags = akCs1CalobTagger.CombinedSecondaryVertexV2BJetTags

akCs1CaloSecondaryVertexNegativeTagInfos = akCs1CalobTagger.SecondaryVertexNegativeTagInfos
akCs1CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCs1CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs1CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCs1CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs1CaloNegativeCombinedSecondaryVertexBJetTags = akCs1CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCs1CaloPositiveCombinedSecondaryVertexBJetTags = akCs1CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCs1CaloSoftPFMuonsTagInfos = akCs1CalobTagger.SoftPFMuonsTagInfos
akCs1CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs1CaloSoftPFMuonBJetTags = akCs1CalobTagger.SoftPFMuonBJetTags
akCs1CaloSoftPFMuonByIP3dBJetTags = akCs1CalobTagger.SoftPFMuonByIP3dBJetTags
akCs1CaloSoftPFMuonByPtBJetTags = akCs1CalobTagger.SoftPFMuonByPtBJetTags
akCs1CaloNegativeSoftPFMuonByPtBJetTags = akCs1CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCs1CaloPositiveSoftPFMuonByPtBJetTags = akCs1CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCs1CaloPatJetFlavourIdLegacy = cms.Sequence(akCs1CaloPatJetPartonAssociationLegacy*akCs1CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs1CaloPatJetFlavourAssociation = akCs1CalobTagger.PatJetFlavourAssociation
#akCs1CaloPatJetFlavourId = cms.Sequence(akCs1CaloPatJetPartons*akCs1CaloPatJetFlavourAssociation)

akCs1CaloJetBtaggingIP       = cms.Sequence(akCs1CaloImpactParameterTagInfos *
            (akCs1CaloTrackCountingHighEffBJetTags +
             akCs1CaloTrackCountingHighPurBJetTags +
             akCs1CaloJetProbabilityBJetTags +
             akCs1CaloJetBProbabilityBJetTags +
             akCs1CaloPositiveOnlyJetProbabilityBJetTags +
             akCs1CaloNegativeOnlyJetProbabilityBJetTags +
             akCs1CaloNegativeTrackCountingHighEffBJetTags +
             akCs1CaloNegativeTrackCountingHighPurBJetTags +
             akCs1CaloNegativeOnlyJetBProbabilityBJetTags +
             akCs1CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCs1CaloJetBtaggingSV = cms.Sequence(akCs1CaloImpactParameterTagInfos
            *
            akCs1CaloSecondaryVertexTagInfos
            * (akCs1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCs1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCs1CaloCombinedSecondaryVertexBJetTags
                +
                akCs1CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCs1CaloJetBtaggingNegSV = cms.Sequence(akCs1CaloImpactParameterTagInfos
            *
            akCs1CaloSecondaryVertexNegativeTagInfos
            * (akCs1CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCs1CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCs1CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCs1CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCs1CaloJetBtaggingMu = cms.Sequence(akCs1CaloSoftPFMuonsTagInfos * (akCs1CaloSoftPFMuonBJetTags
                +
                akCs1CaloSoftPFMuonByIP3dBJetTags
                +
                akCs1CaloSoftPFMuonByPtBJetTags
                +
                akCs1CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCs1CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs1CaloJetBtagging = cms.Sequence(akCs1CaloJetBtaggingIP
            *akCs1CaloJetBtaggingSV
            *akCs1CaloJetBtaggingNegSV
#            *akCs1CaloJetBtaggingMu
            )

akCs1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs1CaloJets"),
        genJetMatch          = cms.InputTag("akCs1Calomatch"),
        genPartonMatch       = cms.InputTag("akCs1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCs1CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs1CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs1CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCs1CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCs1CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs1CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs1CaloJetID"),
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

akCs1CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs1CaloJets"),
           	    R0  = cms.double( 0.1)
)
akCs1CalopatJetsWithBtagging.userData.userFloats.src += ['akCs1CaloNjettiness:tau1','akCs1CaloNjettiness:tau2','akCs1CaloNjettiness:tau3']

akCs1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akCs1Calo"),
                                                             jetName = cms.untracked.string("akCs1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akCs1CaloJetSequence_mc = cms.Sequence(
                                                  #akCs1Caloclean
                                                  #*
                                                  akCs1Calomatch
                                                  *
                                                  akCs1Caloparton
                                                  *
                                                  akCs1Calocorr
                                                  *
                                                  #akCs1CaloJetID
                                                  #*
                                                  akCs1CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCs1CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCs1CaloJetBtagging
                                                  *
                                                  akCs1CaloNjettiness
                                                  *
                                                  akCs1CalopatJetsWithBtagging
                                                  *
                                                  akCs1CaloJetAnalyzer
                                                  )

akCs1CaloJetSequence_data = cms.Sequence(akCs1Calocorr
                                                    *
                                                    #akCs1CaloJetID
                                                    #*
                                                    akCs1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCs1CaloJetBtagging
                                                    *
                                                    akCs1CaloNjettiness 
                                                    *
                                                    akCs1CalopatJetsWithBtagging
                                                    *
                                                    akCs1CaloJetAnalyzer
                                                    )

akCs1CaloJetSequence_jec = cms.Sequence(akCs1CaloJetSequence_mc)
akCs1CaloJetSequence_mb = cms.Sequence(akCs1CaloJetSequence_mc)

akCs1CaloJetSequence = cms.Sequence(akCs1CaloJetSequence_data)

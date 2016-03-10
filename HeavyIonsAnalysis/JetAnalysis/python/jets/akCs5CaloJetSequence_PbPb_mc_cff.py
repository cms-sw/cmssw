

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs5CaloJets"),
    matched = cms.InputTag("ak5HiSignalGenJets"),
    maxDeltaR = 0.5
    )

akCs5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs5CaloJets")
                                                        )

akCs5Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs5CaloJets"),
    payload = "AK5Calo_offline"
    )

akCs5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs5CaloJets'))

#akCs5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiSignalGenJets'))

akCs5CalobTagger = bTaggers("akCs5Calo",0.5)

#create objects locally since they dont load properly otherwise
#akCs5Calomatch = akCs5CalobTagger.match
akCs5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCs5CaloJets"), matched = cms.InputTag("selectedPartons"))
akCs5CaloPatJetFlavourAssociationLegacy = akCs5CalobTagger.PatJetFlavourAssociationLegacy
akCs5CaloPatJetPartons = akCs5CalobTagger.PatJetPartons
akCs5CaloJetTracksAssociatorAtVertex = akCs5CalobTagger.JetTracksAssociatorAtVertex
akCs5CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs5CaloSimpleSecondaryVertexHighEffBJetTags = akCs5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs5CaloSimpleSecondaryVertexHighPurBJetTags = akCs5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs5CaloCombinedSecondaryVertexBJetTags = akCs5CalobTagger.CombinedSecondaryVertexBJetTags
akCs5CaloCombinedSecondaryVertexV2BJetTags = akCs5CalobTagger.CombinedSecondaryVertexV2BJetTags
akCs5CaloJetBProbabilityBJetTags = akCs5CalobTagger.JetBProbabilityBJetTags
akCs5CaloSoftPFMuonByPtBJetTags = akCs5CalobTagger.SoftPFMuonByPtBJetTags
akCs5CaloSoftPFMuonByIP3dBJetTags = akCs5CalobTagger.SoftPFMuonByIP3dBJetTags
akCs5CaloTrackCountingHighEffBJetTags = akCs5CalobTagger.TrackCountingHighEffBJetTags
akCs5CaloTrackCountingHighPurBJetTags = akCs5CalobTagger.TrackCountingHighPurBJetTags
akCs5CaloPatJetPartonAssociationLegacy = akCs5CalobTagger.PatJetPartonAssociationLegacy

akCs5CaloImpactParameterTagInfos = akCs5CalobTagger.ImpactParameterTagInfos
akCs5CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs5CaloJetProbabilityBJetTags = akCs5CalobTagger.JetProbabilityBJetTags
akCs5CaloPositiveOnlyJetProbabilityBJetTags = akCs5CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCs5CaloNegativeOnlyJetProbabilityBJetTags = akCs5CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCs5CaloNegativeTrackCountingHighEffBJetTags = akCs5CalobTagger.NegativeTrackCountingHighEffBJetTags
akCs5CaloNegativeTrackCountingHighPurBJetTags = akCs5CalobTagger.NegativeTrackCountingHighPurBJetTags
akCs5CaloNegativeOnlyJetBProbabilityBJetTags = akCs5CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCs5CaloPositiveOnlyJetBProbabilityBJetTags = akCs5CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCs5CaloSecondaryVertexTagInfos = akCs5CalobTagger.SecondaryVertexTagInfos
akCs5CaloSimpleSecondaryVertexHighEffBJetTags = akCs5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCs5CaloSimpleSecondaryVertexHighPurBJetTags = akCs5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCs5CaloCombinedSecondaryVertexBJetTags = akCs5CalobTagger.CombinedSecondaryVertexBJetTags
akCs5CaloCombinedSecondaryVertexV2BJetTags = akCs5CalobTagger.CombinedSecondaryVertexV2BJetTags

akCs5CaloSecondaryVertexNegativeTagInfos = akCs5CalobTagger.SecondaryVertexNegativeTagInfos
akCs5CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCs5CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs5CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCs5CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs5CaloNegativeCombinedSecondaryVertexBJetTags = akCs5CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCs5CaloPositiveCombinedSecondaryVertexBJetTags = akCs5CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCs5CaloSoftPFMuonsTagInfos = akCs5CalobTagger.SoftPFMuonsTagInfos
akCs5CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs5CaloSoftPFMuonBJetTags = akCs5CalobTagger.SoftPFMuonBJetTags
akCs5CaloSoftPFMuonByIP3dBJetTags = akCs5CalobTagger.SoftPFMuonByIP3dBJetTags
akCs5CaloSoftPFMuonByPtBJetTags = akCs5CalobTagger.SoftPFMuonByPtBJetTags
akCs5CaloNegativeSoftPFMuonByPtBJetTags = akCs5CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCs5CaloPositiveSoftPFMuonByPtBJetTags = akCs5CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCs5CaloPatJetFlavourIdLegacy = cms.Sequence(akCs5CaloPatJetPartonAssociationLegacy*akCs5CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs5CaloPatJetFlavourAssociation = akCs5CalobTagger.PatJetFlavourAssociation
#akCs5CaloPatJetFlavourId = cms.Sequence(akCs5CaloPatJetPartons*akCs5CaloPatJetFlavourAssociation)

akCs5CaloJetBtaggingIP       = cms.Sequence(akCs5CaloImpactParameterTagInfos *
            (akCs5CaloTrackCountingHighEffBJetTags +
             akCs5CaloTrackCountingHighPurBJetTags +
             akCs5CaloJetProbabilityBJetTags +
             akCs5CaloJetBProbabilityBJetTags +
             akCs5CaloPositiveOnlyJetProbabilityBJetTags +
             akCs5CaloNegativeOnlyJetProbabilityBJetTags +
             akCs5CaloNegativeTrackCountingHighEffBJetTags +
             akCs5CaloNegativeTrackCountingHighPurBJetTags +
             akCs5CaloNegativeOnlyJetBProbabilityBJetTags +
             akCs5CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCs5CaloJetBtaggingSV = cms.Sequence(akCs5CaloImpactParameterTagInfos
            *
            akCs5CaloSecondaryVertexTagInfos
            * (akCs5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCs5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCs5CaloCombinedSecondaryVertexBJetTags
                +
                akCs5CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCs5CaloJetBtaggingNegSV = cms.Sequence(akCs5CaloImpactParameterTagInfos
            *
            akCs5CaloSecondaryVertexNegativeTagInfos
            * (akCs5CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCs5CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCs5CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCs5CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCs5CaloJetBtaggingMu = cms.Sequence(akCs5CaloSoftPFMuonsTagInfos * (akCs5CaloSoftPFMuonBJetTags
                +
                akCs5CaloSoftPFMuonByIP3dBJetTags
                +
                akCs5CaloSoftPFMuonByPtBJetTags
                +
                akCs5CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCs5CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs5CaloJetBtagging = cms.Sequence(akCs5CaloJetBtaggingIP
            *akCs5CaloJetBtaggingSV
            *akCs5CaloJetBtaggingNegSV
#            *akCs5CaloJetBtaggingMu
            )

akCs5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs5CaloJets"),
        genJetMatch          = cms.InputTag("akCs5Calomatch"),
        genPartonMatch       = cms.InputTag("akCs5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs5Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCs5CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs5CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs5CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs5CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCs5CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCs5CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs5CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs5CaloJetID"),
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

akCs5CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs5CaloJets"),
           	    R0  = cms.double( 0.5)
)
akCs5CalopatJetsWithBtagging.userData.userFloats.src += ['akCs5CaloNjettiness:tau1','akCs5CaloNjettiness:tau2','akCs5CaloNjettiness:tau3']

akCs5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs5CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akCs5Calo"),
                                                             jetName = cms.untracked.string("akCs5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs5CaloJetSequence_mc = cms.Sequence(
                                                  #akCs5Caloclean
                                                  #*
                                                  akCs5Calomatch
                                                  *
                                                  akCs5Caloparton
                                                  *
                                                  akCs5Calocorr
                                                  *
                                                  #akCs5CaloJetID
                                                  #*
                                                  akCs5CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCs5CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCs5CaloJetBtagging
                                                  *
                                                  akCs5CaloNjettiness
                                                  *
                                                  akCs5CalopatJetsWithBtagging
                                                  *
                                                  akCs5CaloJetAnalyzer
                                                  )

akCs5CaloJetSequence_data = cms.Sequence(akCs5Calocorr
                                                    *
                                                    #akCs5CaloJetID
                                                    #*
                                                    akCs5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCs5CaloJetBtagging
                                                    *
                                                    akCs5CaloNjettiness 
                                                    *
                                                    akCs5CalopatJetsWithBtagging
                                                    *
                                                    akCs5CaloJetAnalyzer
                                                    )

akCs5CaloJetSequence_jec = cms.Sequence(akCs5CaloJetSequence_mc)
akCs5CaloJetSequence_mb = cms.Sequence(akCs5CaloJetSequence_mc)

akCs5CaloJetSequence = cms.Sequence(akCs5CaloJetSequence_mc)

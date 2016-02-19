

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akVs5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs5CaloJets"),
    matched = cms.InputTag("ak5HiGenJets"),
    maxDeltaR = 0.5
    )

akVs5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5CaloJets")
                                                        )

akVs5Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs5CaloJets"),
    payload = "AK5Calo_offline"
    )

akVs5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs5CaloJets'))

#akVs5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJets'))

akVs5CalobTagger = bTaggers("akVs5Calo",0.5)

#create objects locally since they dont load properly otherwise
#akVs5Calomatch = akVs5CalobTagger.match
akVs5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5CaloJets"), matched = cms.InputTag("genParticles"))
akVs5CaloPatJetFlavourAssociationLegacy = akVs5CalobTagger.PatJetFlavourAssociationLegacy
akVs5CaloPatJetPartons = akVs5CalobTagger.PatJetPartons
akVs5CaloJetTracksAssociatorAtVertex = akVs5CalobTagger.JetTracksAssociatorAtVertex
akVs5CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs5CaloSimpleSecondaryVertexHighEffBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5CaloSimpleSecondaryVertexHighPurBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5CaloCombinedSecondaryVertexBJetTags = akVs5CalobTagger.CombinedSecondaryVertexBJetTags
akVs5CaloCombinedSecondaryVertexV2BJetTags = akVs5CalobTagger.CombinedSecondaryVertexV2BJetTags
akVs5CaloJetBProbabilityBJetTags = akVs5CalobTagger.JetBProbabilityBJetTags
akVs5CaloSoftPFMuonByPtBJetTags = akVs5CalobTagger.SoftPFMuonByPtBJetTags
akVs5CaloSoftPFMuonByIP3dBJetTags = akVs5CalobTagger.SoftPFMuonByIP3dBJetTags
akVs5CaloTrackCountingHighEffBJetTags = akVs5CalobTagger.TrackCountingHighEffBJetTags
akVs5CaloTrackCountingHighPurBJetTags = akVs5CalobTagger.TrackCountingHighPurBJetTags
akVs5CaloPatJetPartonAssociationLegacy = akVs5CalobTagger.PatJetPartonAssociationLegacy

akVs5CaloImpactParameterTagInfos = akVs5CalobTagger.ImpactParameterTagInfos
akVs5CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs5CaloJetProbabilityBJetTags = akVs5CalobTagger.JetProbabilityBJetTags
akVs5CaloPositiveOnlyJetProbabilityBJetTags = akVs5CalobTagger.PositiveOnlyJetProbabilityBJetTags
akVs5CaloNegativeOnlyJetProbabilityBJetTags = akVs5CalobTagger.NegativeOnlyJetProbabilityBJetTags
akVs5CaloNegativeTrackCountingHighEffBJetTags = akVs5CalobTagger.NegativeTrackCountingHighEffBJetTags
akVs5CaloNegativeTrackCountingHighPurBJetTags = akVs5CalobTagger.NegativeTrackCountingHighPurBJetTags
akVs5CaloNegativeOnlyJetBProbabilityBJetTags = akVs5CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akVs5CaloPositiveOnlyJetBProbabilityBJetTags = akVs5CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akVs5CaloSecondaryVertexTagInfos = akVs5CalobTagger.SecondaryVertexTagInfos
akVs5CaloSimpleSecondaryVertexHighEffBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5CaloSimpleSecondaryVertexHighPurBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5CaloCombinedSecondaryVertexBJetTags = akVs5CalobTagger.CombinedSecondaryVertexBJetTags
akVs5CaloCombinedSecondaryVertexV2BJetTags = akVs5CalobTagger.CombinedSecondaryVertexV2BJetTags

akVs5CaloSecondaryVertexNegativeTagInfos = akVs5CalobTagger.SecondaryVertexNegativeTagInfos
akVs5CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akVs5CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs5CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akVs5CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs5CaloNegativeCombinedSecondaryVertexBJetTags = akVs5CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akVs5CaloPositiveCombinedSecondaryVertexBJetTags = akVs5CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akVs5CaloSoftPFMuonsTagInfos = akVs5CalobTagger.SoftPFMuonsTagInfos
akVs5CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs5CaloSoftPFMuonBJetTags = akVs5CalobTagger.SoftPFMuonBJetTags
akVs5CaloSoftPFMuonByIP3dBJetTags = akVs5CalobTagger.SoftPFMuonByIP3dBJetTags
akVs5CaloSoftPFMuonByPtBJetTags = akVs5CalobTagger.SoftPFMuonByPtBJetTags
akVs5CaloNegativeSoftPFMuonByPtBJetTags = akVs5CalobTagger.NegativeSoftPFMuonByPtBJetTags
akVs5CaloPositiveSoftPFMuonByPtBJetTags = akVs5CalobTagger.PositiveSoftPFMuonByPtBJetTags
akVs5CaloPatJetFlavourIdLegacy = cms.Sequence(akVs5CaloPatJetPartonAssociationLegacy*akVs5CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs5CaloPatJetFlavourAssociation = akVs5CalobTagger.PatJetFlavourAssociation
#akVs5CaloPatJetFlavourId = cms.Sequence(akVs5CaloPatJetPartons*akVs5CaloPatJetFlavourAssociation)

akVs5CaloJetBtaggingIP       = cms.Sequence(akVs5CaloImpactParameterTagInfos *
            (akVs5CaloTrackCountingHighEffBJetTags +
             akVs5CaloTrackCountingHighPurBJetTags +
             akVs5CaloJetProbabilityBJetTags +
             akVs5CaloJetBProbabilityBJetTags +
             akVs5CaloPositiveOnlyJetProbabilityBJetTags +
             akVs5CaloNegativeOnlyJetProbabilityBJetTags +
             akVs5CaloNegativeTrackCountingHighEffBJetTags +
             akVs5CaloNegativeTrackCountingHighPurBJetTags +
             akVs5CaloNegativeOnlyJetBProbabilityBJetTags +
             akVs5CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs5CaloJetBtaggingSV = cms.Sequence(akVs5CaloImpactParameterTagInfos
            *
            akVs5CaloSecondaryVertexTagInfos
            * (akVs5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs5CaloCombinedSecondaryVertexBJetTags
                +
                akVs5CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akVs5CaloJetBtaggingNegSV = cms.Sequence(akVs5CaloImpactParameterTagInfos
            *
            akVs5CaloSecondaryVertexNegativeTagInfos
            * (akVs5CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs5CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs5CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akVs5CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs5CaloJetBtaggingMu = cms.Sequence(akVs5CaloSoftPFMuonsTagInfos * (akVs5CaloSoftPFMuonBJetTags
                +
                akVs5CaloSoftPFMuonByIP3dBJetTags
                +
                akVs5CaloSoftPFMuonByPtBJetTags
                +
                akVs5CaloNegativeSoftPFMuonByPtBJetTags
                +
                akVs5CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs5CaloJetBtagging = cms.Sequence(akVs5CaloJetBtaggingIP
            *akVs5CaloJetBtaggingSV
            *akVs5CaloJetBtaggingNegSV
#            *akVs5CaloJetBtaggingMu
            )

akVs5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs5CaloJets"),
        genJetMatch          = cms.InputTag("akVs5Calomatch"),
        genPartonMatch       = cms.InputTag("akVs5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs5CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs5CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs5CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs5CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs5CaloJetProbabilityBJetTags"),
            #cms.InputTag("akVs5CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs5CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs5CaloJetID"),
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

akVs5CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akVs5CaloJets"),
           	    R0  = cms.double( 0.5)
)
akVs5CalopatJetsWithBtagging.userData.userFloats.src += ['akVs5CaloNjettiness:tau1','akVs5CaloNjettiness:tau2','akVs5CaloNjettiness:tau3']

akVs5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5CalopatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akVs5Calo"),
                                                             jetName = cms.untracked.string("akVs5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akVs5CaloJetSequence_mc = cms.Sequence(
                                                  #akVs5Caloclean
                                                  #*
                                                  akVs5Calomatch
                                                  *
                                                  akVs5Caloparton
                                                  *
                                                  akVs5Calocorr
                                                  *
                                                  #akVs5CaloJetID
                                                  #*
                                                  akVs5CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akVs5CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs5CaloJetBtagging
                                                  *
                                                  akVs5CaloNjettiness
                                                  *
                                                  akVs5CalopatJetsWithBtagging
                                                  *
                                                  akVs5CaloJetAnalyzer
                                                  )

akVs5CaloJetSequence_data = cms.Sequence(akVs5Calocorr
                                                    *
                                                    #akVs5CaloJetID
                                                    #*
                                                    akVs5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs5CaloJetBtagging
                                                    *
                                                    akVs5CaloNjettiness 
                                                    *
                                                    akVs5CalopatJetsWithBtagging
                                                    *
                                                    akVs5CaloJetAnalyzer
                                                    )

akVs5CaloJetSequence_jec = cms.Sequence(akVs5CaloJetSequence_mc)
akVs5CaloJetSequence_mix = cms.Sequence(akVs5CaloJetSequence_mc)

akVs5CaloJetSequence = cms.Sequence(akVs5CaloJetSequence_mc)

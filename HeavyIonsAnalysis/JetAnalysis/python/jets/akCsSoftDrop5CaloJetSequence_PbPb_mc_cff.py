

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop5CaloJets"),
    matched = cms.InputTag("ak5HiSignalGenJets"),
    maxDeltaR = 0.5
    )

akCsSoftDrop5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop5CaloJets")
                                                        )

akCsSoftDrop5Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop5CaloJets"),
    payload = "AK5Calo_offline"
    )

akCsSoftDrop5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop5CaloJets'))

#akCsSoftDrop5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiSignalGenJets'))

akCsSoftDrop5CalobTagger = bTaggers("akCsSoftDrop5Calo",0.5)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop5Calomatch = akCsSoftDrop5CalobTagger.match
akCsSoftDrop5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop5CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop5CaloPatJetFlavourAssociationLegacy = akCsSoftDrop5CalobTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop5CaloPatJetPartons = akCsSoftDrop5CalobTagger.PatJetPartons
akCsSoftDrop5CaloJetTracksAssociatorAtVertex = akCsSoftDrop5CalobTagger.JetTracksAssociatorAtVertex
akCsSoftDrop5CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop5CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop5CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop5CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop5CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop5CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop5CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop5CaloJetBProbabilityBJetTags = akCsSoftDrop5CalobTagger.JetBProbabilityBJetTags
akCsSoftDrop5CaloSoftPFMuonByPtBJetTags = akCsSoftDrop5CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop5CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop5CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop5CaloTrackCountingHighEffBJetTags = akCsSoftDrop5CalobTagger.TrackCountingHighEffBJetTags
akCsSoftDrop5CaloTrackCountingHighPurBJetTags = akCsSoftDrop5CalobTagger.TrackCountingHighPurBJetTags
akCsSoftDrop5CaloPatJetPartonAssociationLegacy = akCsSoftDrop5CalobTagger.PatJetPartonAssociationLegacy

akCsSoftDrop5CaloImpactParameterTagInfos = akCsSoftDrop5CalobTagger.ImpactParameterTagInfos
akCsSoftDrop5CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop5CaloJetProbabilityBJetTags = akCsSoftDrop5CalobTagger.JetProbabilityBJetTags
akCsSoftDrop5CaloPositiveOnlyJetProbabilityBJetTags = akCsSoftDrop5CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCsSoftDrop5CaloNegativeOnlyJetProbabilityBJetTags = akCsSoftDrop5CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCsSoftDrop5CaloNegativeTrackCountingHighEffBJetTags = akCsSoftDrop5CalobTagger.NegativeTrackCountingHighEffBJetTags
akCsSoftDrop5CaloNegativeTrackCountingHighPurBJetTags = akCsSoftDrop5CalobTagger.NegativeTrackCountingHighPurBJetTags
akCsSoftDrop5CaloNegativeOnlyJetBProbabilityBJetTags = akCsSoftDrop5CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCsSoftDrop5CaloPositiveOnlyJetBProbabilityBJetTags = akCsSoftDrop5CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCsSoftDrop5CaloSecondaryVertexTagInfos = akCsSoftDrop5CalobTagger.SecondaryVertexTagInfos
akCsSoftDrop5CaloSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop5CaloSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop5CaloCombinedSecondaryVertexBJetTags = akCsSoftDrop5CalobTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop5CaloCombinedSecondaryVertexV2BJetTags = akCsSoftDrop5CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop5CaloSecondaryVertexNegativeTagInfos = akCsSoftDrop5CalobTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop5CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop5CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop5CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop5CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop5CaloNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop5CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop5CaloPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop5CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCsSoftDrop5CaloSoftPFMuonsTagInfos = akCsSoftDrop5CalobTagger.SoftPFMuonsTagInfos
akCsSoftDrop5CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop5CaloSoftPFMuonBJetTags = akCsSoftDrop5CalobTagger.SoftPFMuonBJetTags
akCsSoftDrop5CaloSoftPFMuonByIP3dBJetTags = akCsSoftDrop5CalobTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop5CaloSoftPFMuonByPtBJetTags = akCsSoftDrop5CalobTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop5CaloNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop5CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop5CaloPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop5CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop5CaloPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop5CaloPatJetPartonAssociationLegacy*akCsSoftDrop5CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop5CaloPatJetFlavourAssociation = akCsSoftDrop5CalobTagger.PatJetFlavourAssociation
#akCsSoftDrop5CaloPatJetFlavourId = cms.Sequence(akCsSoftDrop5CaloPatJetPartons*akCsSoftDrop5CaloPatJetFlavourAssociation)

akCsSoftDrop5CaloJetBtaggingIP       = cms.Sequence(akCsSoftDrop5CaloImpactParameterTagInfos *
            (akCsSoftDrop5CaloTrackCountingHighEffBJetTags +
             akCsSoftDrop5CaloTrackCountingHighPurBJetTags +
             akCsSoftDrop5CaloJetProbabilityBJetTags +
             akCsSoftDrop5CaloJetBProbabilityBJetTags +
             akCsSoftDrop5CaloPositiveOnlyJetProbabilityBJetTags +
             akCsSoftDrop5CaloNegativeOnlyJetProbabilityBJetTags +
             akCsSoftDrop5CaloNegativeTrackCountingHighEffBJetTags +
             akCsSoftDrop5CaloNegativeTrackCountingHighPurBJetTags +
             akCsSoftDrop5CaloNegativeOnlyJetBProbabilityBJetTags +
             akCsSoftDrop5CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCsSoftDrop5CaloJetBtaggingSV = cms.Sequence(akCsSoftDrop5CaloImpactParameterTagInfos
            *
            akCsSoftDrop5CaloSecondaryVertexTagInfos
            * (akCsSoftDrop5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCsSoftDrop5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCsSoftDrop5CaloCombinedSecondaryVertexBJetTags
                +
                akCsSoftDrop5CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop5CaloJetBtaggingNegSV = cms.Sequence(akCsSoftDrop5CaloImpactParameterTagInfos
            *
            akCsSoftDrop5CaloSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop5CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCsSoftDrop5CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCsSoftDrop5CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCsSoftDrop5CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCsSoftDrop5CaloJetBtaggingMu = cms.Sequence(akCsSoftDrop5CaloSoftPFMuonsTagInfos * (akCsSoftDrop5CaloSoftPFMuonBJetTags
                +
                akCsSoftDrop5CaloSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop5CaloSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop5CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop5CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop5CaloJetBtagging = cms.Sequence(akCsSoftDrop5CaloJetBtaggingIP
            *akCsSoftDrop5CaloJetBtaggingSV
            *akCsSoftDrop5CaloJetBtaggingNegSV
#            *akCsSoftDrop5CaloJetBtaggingMu
            )

akCsSoftDrop5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop5CaloJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop5Calomatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop5Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop5CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop5CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop5CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop5CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop5CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop5CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop5CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop5CaloJetID"),
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

akCsSoftDrop5CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop5CaloJets"),
           	    R0  = cms.double( 0.5)
)
akCsSoftDrop5CalopatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop5CaloNjettiness:tau1','akCsSoftDrop5CaloNjettiness:tau2','akCsSoftDrop5CaloNjettiness:tau3']

akCsSoftDrop5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop5CalopatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop5Calo"),
                                                             jetName = cms.untracked.string("akCsSoftDrop5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop5CaloJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop5Caloclean
                                                  #*
                                                  akCsSoftDrop5Calomatch
                                                  *
                                                  akCsSoftDrop5Caloparton
                                                  *
                                                  akCsSoftDrop5Calocorr
                                                  *
                                                  #akCsSoftDrop5CaloJetID
                                                  #*
                                                  akCsSoftDrop5CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop5CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop5CaloJetBtagging
                                                  *
                                                  akCsSoftDrop5CaloNjettiness
                                                  *
                                                  akCsSoftDrop5CalopatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop5CaloJetAnalyzer
                                                  )

akCsSoftDrop5CaloJetSequence_data = cms.Sequence(akCsSoftDrop5Calocorr
                                                    *
                                                    #akCsSoftDrop5CaloJetID
                                                    #*
                                                    akCsSoftDrop5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop5CaloJetBtagging
                                                    *
                                                    akCsSoftDrop5CaloNjettiness 
                                                    *
                                                    akCsSoftDrop5CalopatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop5CaloJetAnalyzer
                                                    )

akCsSoftDrop5CaloJetSequence_jec = cms.Sequence(akCsSoftDrop5CaloJetSequence_mc)
akCsSoftDrop5CaloJetSequence_mb = cms.Sequence(akCsSoftDrop5CaloJetSequence_mc)

akCsSoftDrop5CaloJetSequence = cms.Sequence(akCsSoftDrop5CaloJetSequence_mc)

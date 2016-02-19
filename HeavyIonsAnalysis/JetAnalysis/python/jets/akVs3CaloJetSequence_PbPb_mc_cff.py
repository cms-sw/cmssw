

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akVs3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs3CaloJets"),
    matched = cms.InputTag("ak3HiGenJets"),
    maxDeltaR = 0.3
    )

akVs3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3CaloJets")
                                                        )

akVs3Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs3CaloJets"),
    payload = "AK3Calo_offline"
    )

akVs3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs3CaloJets'))

#akVs3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJets'))

akVs3CalobTagger = bTaggers("akVs3Calo",0.3)

#create objects locally since they dont load properly otherwise
#akVs3Calomatch = akVs3CalobTagger.match
akVs3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3CaloJets"), matched = cms.InputTag("genParticles"))
akVs3CaloPatJetFlavourAssociationLegacy = akVs3CalobTagger.PatJetFlavourAssociationLegacy
akVs3CaloPatJetPartons = akVs3CalobTagger.PatJetPartons
akVs3CaloJetTracksAssociatorAtVertex = akVs3CalobTagger.JetTracksAssociatorAtVertex
akVs3CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs3CaloSimpleSecondaryVertexHighEffBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3CaloSimpleSecondaryVertexHighPurBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3CaloCombinedSecondaryVertexBJetTags = akVs3CalobTagger.CombinedSecondaryVertexBJetTags
akVs3CaloCombinedSecondaryVertexV2BJetTags = akVs3CalobTagger.CombinedSecondaryVertexV2BJetTags
akVs3CaloJetBProbabilityBJetTags = akVs3CalobTagger.JetBProbabilityBJetTags
akVs3CaloSoftPFMuonByPtBJetTags = akVs3CalobTagger.SoftPFMuonByPtBJetTags
akVs3CaloSoftPFMuonByIP3dBJetTags = akVs3CalobTagger.SoftPFMuonByIP3dBJetTags
akVs3CaloTrackCountingHighEffBJetTags = akVs3CalobTagger.TrackCountingHighEffBJetTags
akVs3CaloTrackCountingHighPurBJetTags = akVs3CalobTagger.TrackCountingHighPurBJetTags
akVs3CaloPatJetPartonAssociationLegacy = akVs3CalobTagger.PatJetPartonAssociationLegacy

akVs3CaloImpactParameterTagInfos = akVs3CalobTagger.ImpactParameterTagInfos
akVs3CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs3CaloJetProbabilityBJetTags = akVs3CalobTagger.JetProbabilityBJetTags
akVs3CaloPositiveOnlyJetProbabilityBJetTags = akVs3CalobTagger.PositiveOnlyJetProbabilityBJetTags
akVs3CaloNegativeOnlyJetProbabilityBJetTags = akVs3CalobTagger.NegativeOnlyJetProbabilityBJetTags
akVs3CaloNegativeTrackCountingHighEffBJetTags = akVs3CalobTagger.NegativeTrackCountingHighEffBJetTags
akVs3CaloNegativeTrackCountingHighPurBJetTags = akVs3CalobTagger.NegativeTrackCountingHighPurBJetTags
akVs3CaloNegativeOnlyJetBProbabilityBJetTags = akVs3CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akVs3CaloPositiveOnlyJetBProbabilityBJetTags = akVs3CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akVs3CaloSecondaryVertexTagInfos = akVs3CalobTagger.SecondaryVertexTagInfos
akVs3CaloSimpleSecondaryVertexHighEffBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3CaloSimpleSecondaryVertexHighPurBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3CaloCombinedSecondaryVertexBJetTags = akVs3CalobTagger.CombinedSecondaryVertexBJetTags
akVs3CaloCombinedSecondaryVertexV2BJetTags = akVs3CalobTagger.CombinedSecondaryVertexV2BJetTags

akVs3CaloSecondaryVertexNegativeTagInfos = akVs3CalobTagger.SecondaryVertexNegativeTagInfos
akVs3CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akVs3CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs3CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akVs3CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs3CaloNegativeCombinedSecondaryVertexBJetTags = akVs3CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akVs3CaloPositiveCombinedSecondaryVertexBJetTags = akVs3CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akVs3CaloSoftPFMuonsTagInfos = akVs3CalobTagger.SoftPFMuonsTagInfos
akVs3CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs3CaloSoftPFMuonBJetTags = akVs3CalobTagger.SoftPFMuonBJetTags
akVs3CaloSoftPFMuonByIP3dBJetTags = akVs3CalobTagger.SoftPFMuonByIP3dBJetTags
akVs3CaloSoftPFMuonByPtBJetTags = akVs3CalobTagger.SoftPFMuonByPtBJetTags
akVs3CaloNegativeSoftPFMuonByPtBJetTags = akVs3CalobTagger.NegativeSoftPFMuonByPtBJetTags
akVs3CaloPositiveSoftPFMuonByPtBJetTags = akVs3CalobTagger.PositiveSoftPFMuonByPtBJetTags
akVs3CaloPatJetFlavourIdLegacy = cms.Sequence(akVs3CaloPatJetPartonAssociationLegacy*akVs3CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs3CaloPatJetFlavourAssociation = akVs3CalobTagger.PatJetFlavourAssociation
#akVs3CaloPatJetFlavourId = cms.Sequence(akVs3CaloPatJetPartons*akVs3CaloPatJetFlavourAssociation)

akVs3CaloJetBtaggingIP       = cms.Sequence(akVs3CaloImpactParameterTagInfos *
            (akVs3CaloTrackCountingHighEffBJetTags +
             akVs3CaloTrackCountingHighPurBJetTags +
             akVs3CaloJetProbabilityBJetTags +
             akVs3CaloJetBProbabilityBJetTags +
             akVs3CaloPositiveOnlyJetProbabilityBJetTags +
             akVs3CaloNegativeOnlyJetProbabilityBJetTags +
             akVs3CaloNegativeTrackCountingHighEffBJetTags +
             akVs3CaloNegativeTrackCountingHighPurBJetTags +
             akVs3CaloNegativeOnlyJetBProbabilityBJetTags +
             akVs3CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs3CaloJetBtaggingSV = cms.Sequence(akVs3CaloImpactParameterTagInfos
            *
            akVs3CaloSecondaryVertexTagInfos
            * (akVs3CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs3CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs3CaloCombinedSecondaryVertexBJetTags
                +
                akVs3CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akVs3CaloJetBtaggingNegSV = cms.Sequence(akVs3CaloImpactParameterTagInfos
            *
            akVs3CaloSecondaryVertexNegativeTagInfos
            * (akVs3CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs3CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs3CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akVs3CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs3CaloJetBtaggingMu = cms.Sequence(akVs3CaloSoftPFMuonsTagInfos * (akVs3CaloSoftPFMuonBJetTags
                +
                akVs3CaloSoftPFMuonByIP3dBJetTags
                +
                akVs3CaloSoftPFMuonByPtBJetTags
                +
                akVs3CaloNegativeSoftPFMuonByPtBJetTags
                +
                akVs3CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs3CaloJetBtagging = cms.Sequence(akVs3CaloJetBtaggingIP
            *akVs3CaloJetBtaggingSV
            *akVs3CaloJetBtaggingNegSV
#            *akVs3CaloJetBtaggingMu
            )

akVs3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs3CaloJets"),
        genJetMatch          = cms.InputTag("akVs3Calomatch"),
        genPartonMatch       = cms.InputTag("akVs3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs3Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs3CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs3CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs3CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs3CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs3CaloJetProbabilityBJetTags"),
            #cms.InputTag("akVs3CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs3CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs3CaloJetID"),
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

akVs3CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akVs3CaloJets"),
           	    R0  = cms.double( 0.3)
)
akVs3CalopatJetsWithBtagging.userData.userFloats.src += ['akVs3CaloNjettiness:tau1','akVs3CaloNjettiness:tau2','akVs3CaloNjettiness:tau3']

akVs3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akVs3Calo"),
                                                             jetName = cms.untracked.string("akVs3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akVs3CaloJetSequence_mc = cms.Sequence(
                                                  #akVs3Caloclean
                                                  #*
                                                  akVs3Calomatch
                                                  *
                                                  akVs3Caloparton
                                                  *
                                                  akVs3Calocorr
                                                  *
                                                  #akVs3CaloJetID
                                                  #*
                                                  akVs3CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akVs3CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs3CaloJetBtagging
                                                  *
                                                  akVs3CaloNjettiness
                                                  *
                                                  akVs3CalopatJetsWithBtagging
                                                  *
                                                  akVs3CaloJetAnalyzer
                                                  )

akVs3CaloJetSequence_data = cms.Sequence(akVs3Calocorr
                                                    *
                                                    #akVs3CaloJetID
                                                    #*
                                                    akVs3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs3CaloJetBtagging
                                                    *
                                                    akVs3CaloNjettiness 
                                                    *
                                                    akVs3CalopatJetsWithBtagging
                                                    *
                                                    akVs3CaloJetAnalyzer
                                                    )

akVs3CaloJetSequence_jec = cms.Sequence(akVs3CaloJetSequence_mc)
akVs3CaloJetSequence_mix = cms.Sequence(akVs3CaloJetSequence_mc)

akVs3CaloJetSequence = cms.Sequence(akVs3CaloJetSequence_mc)

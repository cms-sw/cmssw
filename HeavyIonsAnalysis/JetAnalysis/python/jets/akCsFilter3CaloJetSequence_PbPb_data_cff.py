

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter3CaloJets"),
    matched = cms.InputTag("ak3HiSignalGenJets"),
    maxDeltaR = 0.3
    )

akCsFilter3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter3CaloJets")
                                                        )

akCsFilter3Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter3CaloJets"),
    payload = "AK3Calo_offline"
    )

akCsFilter3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter3CaloJets'))

#akCsFilter3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiSignalGenJets'))

akCsFilter3CalobTagger = bTaggers("akCsFilter3Calo",0.3)

#create objects locally since they dont load properly otherwise
#akCsFilter3Calomatch = akCsFilter3CalobTagger.match
akCsFilter3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter3CaloJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter3CaloPatJetFlavourAssociationLegacy = akCsFilter3CalobTagger.PatJetFlavourAssociationLegacy
akCsFilter3CaloPatJetPartons = akCsFilter3CalobTagger.PatJetPartons
akCsFilter3CaloJetTracksAssociatorAtVertex = akCsFilter3CalobTagger.JetTracksAssociatorAtVertex
akCsFilter3CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter3CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter3CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter3CaloCombinedSecondaryVertexBJetTags = akCsFilter3CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter3CaloCombinedSecondaryVertexV2BJetTags = akCsFilter3CalobTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter3CaloJetBProbabilityBJetTags = akCsFilter3CalobTagger.JetBProbabilityBJetTags
akCsFilter3CaloSoftPFMuonByPtBJetTags = akCsFilter3CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter3CaloSoftPFMuonByIP3dBJetTags = akCsFilter3CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter3CaloTrackCountingHighEffBJetTags = akCsFilter3CalobTagger.TrackCountingHighEffBJetTags
akCsFilter3CaloTrackCountingHighPurBJetTags = akCsFilter3CalobTagger.TrackCountingHighPurBJetTags
akCsFilter3CaloPatJetPartonAssociationLegacy = akCsFilter3CalobTagger.PatJetPartonAssociationLegacy

akCsFilter3CaloImpactParameterTagInfos = akCsFilter3CalobTagger.ImpactParameterTagInfos
akCsFilter3CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter3CaloJetProbabilityBJetTags = akCsFilter3CalobTagger.JetProbabilityBJetTags
akCsFilter3CaloPositiveOnlyJetProbabilityBJetTags = akCsFilter3CalobTagger.PositiveOnlyJetProbabilityBJetTags
akCsFilter3CaloNegativeOnlyJetProbabilityBJetTags = akCsFilter3CalobTagger.NegativeOnlyJetProbabilityBJetTags
akCsFilter3CaloNegativeTrackCountingHighEffBJetTags = akCsFilter3CalobTagger.NegativeTrackCountingHighEffBJetTags
akCsFilter3CaloNegativeTrackCountingHighPurBJetTags = akCsFilter3CalobTagger.NegativeTrackCountingHighPurBJetTags
akCsFilter3CaloNegativeOnlyJetBProbabilityBJetTags = akCsFilter3CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akCsFilter3CaloPositiveOnlyJetBProbabilityBJetTags = akCsFilter3CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akCsFilter3CaloSecondaryVertexTagInfos = akCsFilter3CalobTagger.SecondaryVertexTagInfos
akCsFilter3CaloSimpleSecondaryVertexHighEffBJetTags = akCsFilter3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter3CaloSimpleSecondaryVertexHighPurBJetTags = akCsFilter3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter3CaloCombinedSecondaryVertexBJetTags = akCsFilter3CalobTagger.CombinedSecondaryVertexBJetTags
akCsFilter3CaloCombinedSecondaryVertexV2BJetTags = akCsFilter3CalobTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter3CaloSecondaryVertexNegativeTagInfos = akCsFilter3CalobTagger.SecondaryVertexNegativeTagInfos
akCsFilter3CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter3CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter3CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter3CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter3CaloNegativeCombinedSecondaryVertexBJetTags = akCsFilter3CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter3CaloPositiveCombinedSecondaryVertexBJetTags = akCsFilter3CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akCsFilter3CaloSoftPFMuonsTagInfos = akCsFilter3CalobTagger.SoftPFMuonsTagInfos
akCsFilter3CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter3CaloSoftPFMuonBJetTags = akCsFilter3CalobTagger.SoftPFMuonBJetTags
akCsFilter3CaloSoftPFMuonByIP3dBJetTags = akCsFilter3CalobTagger.SoftPFMuonByIP3dBJetTags
akCsFilter3CaloSoftPFMuonByPtBJetTags = akCsFilter3CalobTagger.SoftPFMuonByPtBJetTags
akCsFilter3CaloNegativeSoftPFMuonByPtBJetTags = akCsFilter3CalobTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter3CaloPositiveSoftPFMuonByPtBJetTags = akCsFilter3CalobTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter3CaloPatJetFlavourIdLegacy = cms.Sequence(akCsFilter3CaloPatJetPartonAssociationLegacy*akCsFilter3CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter3CaloPatJetFlavourAssociation = akCsFilter3CalobTagger.PatJetFlavourAssociation
#akCsFilter3CaloPatJetFlavourId = cms.Sequence(akCsFilter3CaloPatJetPartons*akCsFilter3CaloPatJetFlavourAssociation)

akCsFilter3CaloJetBtaggingIP       = cms.Sequence(akCsFilter3CaloImpactParameterTagInfos *
            (akCsFilter3CaloTrackCountingHighEffBJetTags +
             akCsFilter3CaloTrackCountingHighPurBJetTags +
             akCsFilter3CaloJetProbabilityBJetTags +
             akCsFilter3CaloJetBProbabilityBJetTags +
             akCsFilter3CaloPositiveOnlyJetProbabilityBJetTags +
             akCsFilter3CaloNegativeOnlyJetProbabilityBJetTags +
             akCsFilter3CaloNegativeTrackCountingHighEffBJetTags +
             akCsFilter3CaloNegativeTrackCountingHighPurBJetTags +
             akCsFilter3CaloNegativeOnlyJetBProbabilityBJetTags +
             akCsFilter3CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCsFilter3CaloJetBtaggingSV = cms.Sequence(akCsFilter3CaloImpactParameterTagInfos
            *
            akCsFilter3CaloSecondaryVertexTagInfos
            * (akCsFilter3CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akCsFilter3CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akCsFilter3CaloCombinedSecondaryVertexBJetTags
                +
                akCsFilter3CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter3CaloJetBtaggingNegSV = cms.Sequence(akCsFilter3CaloImpactParameterTagInfos
            *
            akCsFilter3CaloSecondaryVertexNegativeTagInfos
            * (akCsFilter3CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCsFilter3CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCsFilter3CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akCsFilter3CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCsFilter3CaloJetBtaggingMu = cms.Sequence(akCsFilter3CaloSoftPFMuonsTagInfos * (akCsFilter3CaloSoftPFMuonBJetTags
                +
                akCsFilter3CaloSoftPFMuonByIP3dBJetTags
                +
                akCsFilter3CaloSoftPFMuonByPtBJetTags
                +
                akCsFilter3CaloNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter3CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter3CaloJetBtagging = cms.Sequence(akCsFilter3CaloJetBtaggingIP
            *akCsFilter3CaloJetBtaggingSV
            *akCsFilter3CaloJetBtaggingNegSV
#            *akCsFilter3CaloJetBtaggingMu
            )

akCsFilter3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter3CaloJets"),
        genJetMatch          = cms.InputTag("akCsFilter3Calomatch"),
        genPartonMatch       = cms.InputTag("akCsFilter3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter3Calocorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter3CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter3CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter3CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter3CaloJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter3CaloJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter3CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter3CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter3CaloJetID"),
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

akCsFilter3CaloNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter3CaloJets"),
           	    R0  = cms.double( 0.3)
)
akCsFilter3CalopatJetsWithBtagging.userData.userFloats.src += ['akCsFilter3CaloNjettiness:tau1','akCsFilter3CaloNjettiness:tau2','akCsFilter3CaloNjettiness:tau3']

akCsFilter3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akCsFilter3Calo"),
                                                             jetName = cms.untracked.string("akCsFilter3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter3CaloJetSequence_mc = cms.Sequence(
                                                  #akCsFilter3Caloclean
                                                  #*
                                                  akCsFilter3Calomatch
                                                  *
                                                  akCsFilter3Caloparton
                                                  *
                                                  akCsFilter3Calocorr
                                                  *
                                                  #akCsFilter3CaloJetID
                                                  #*
                                                  akCsFilter3CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter3CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter3CaloJetBtagging
                                                  *
                                                  akCsFilter3CaloNjettiness
                                                  *
                                                  akCsFilter3CalopatJetsWithBtagging
                                                  *
                                                  akCsFilter3CaloJetAnalyzer
                                                  )

akCsFilter3CaloJetSequence_data = cms.Sequence(akCsFilter3Calocorr
                                                    *
                                                    #akCsFilter3CaloJetID
                                                    #*
                                                    akCsFilter3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter3CaloJetBtagging
                                                    *
                                                    akCsFilter3CaloNjettiness 
                                                    *
                                                    akCsFilter3CalopatJetsWithBtagging
                                                    *
                                                    akCsFilter3CaloJetAnalyzer
                                                    )

akCsFilter3CaloJetSequence_jec = cms.Sequence(akCsFilter3CaloJetSequence_mc)
akCsFilter3CaloJetSequence_mb = cms.Sequence(akCsFilter3CaloJetSequence_mc)

akCsFilter3CaloJetSequence = cms.Sequence(akCsFilter3CaloJetSequence_data)



import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs4CaloJets"),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = 0.4
    )

akVs4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4CaloJets")
                                                        )

akVs4Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs4CaloJets"),
    payload = "AK4Calo_offline"
    )

akVs4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs4CaloJets'))

#akVs4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4GenJets'))

akVs4CalobTagger = bTaggers("akVs4Calo",0.4)

#create objects locally since they dont load properly otherwise
#akVs4Calomatch = akVs4CalobTagger.match
akVs4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4CaloJets"), matched = cms.InputTag("genParticles"))
akVs4CaloPatJetFlavourAssociationLegacy = akVs4CalobTagger.PatJetFlavourAssociationLegacy
akVs4CaloPatJetPartons = akVs4CalobTagger.PatJetPartons
akVs4CaloJetTracksAssociatorAtVertex = akVs4CalobTagger.JetTracksAssociatorAtVertex
akVs4CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs4CaloSimpleSecondaryVertexHighEffBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4CaloSimpleSecondaryVertexHighPurBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4CaloCombinedSecondaryVertexBJetTags = akVs4CalobTagger.CombinedSecondaryVertexBJetTags
akVs4CaloCombinedSecondaryVertexV2BJetTags = akVs4CalobTagger.CombinedSecondaryVertexV2BJetTags
akVs4CaloJetBProbabilityBJetTags = akVs4CalobTagger.JetBProbabilityBJetTags
akVs4CaloSoftPFMuonByPtBJetTags = akVs4CalobTagger.SoftPFMuonByPtBJetTags
akVs4CaloSoftPFMuonByIP3dBJetTags = akVs4CalobTagger.SoftPFMuonByIP3dBJetTags
akVs4CaloTrackCountingHighEffBJetTags = akVs4CalobTagger.TrackCountingHighEffBJetTags
akVs4CaloTrackCountingHighPurBJetTags = akVs4CalobTagger.TrackCountingHighPurBJetTags
akVs4CaloPatJetPartonAssociationLegacy = akVs4CalobTagger.PatJetPartonAssociationLegacy

akVs4CaloImpactParameterTagInfos = akVs4CalobTagger.ImpactParameterTagInfos
akVs4CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs4CaloJetProbabilityBJetTags = akVs4CalobTagger.JetProbabilityBJetTags
akVs4CaloPositiveOnlyJetProbabilityBJetTags = akVs4CalobTagger.PositiveOnlyJetProbabilityBJetTags
akVs4CaloNegativeOnlyJetProbabilityBJetTags = akVs4CalobTagger.NegativeOnlyJetProbabilityBJetTags
akVs4CaloNegativeTrackCountingHighEffBJetTags = akVs4CalobTagger.NegativeTrackCountingHighEffBJetTags
akVs4CaloNegativeTrackCountingHighPurBJetTags = akVs4CalobTagger.NegativeTrackCountingHighPurBJetTags
akVs4CaloNegativeOnlyJetBProbabilityBJetTags = akVs4CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akVs4CaloPositiveOnlyJetBProbabilityBJetTags = akVs4CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akVs4CaloSecondaryVertexTagInfos = akVs4CalobTagger.SecondaryVertexTagInfos
akVs4CaloSimpleSecondaryVertexHighEffBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4CaloSimpleSecondaryVertexHighPurBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4CaloCombinedSecondaryVertexBJetTags = akVs4CalobTagger.CombinedSecondaryVertexBJetTags
akVs4CaloCombinedSecondaryVertexV2BJetTags = akVs4CalobTagger.CombinedSecondaryVertexV2BJetTags

akVs4CaloSecondaryVertexNegativeTagInfos = akVs4CalobTagger.SecondaryVertexNegativeTagInfos
akVs4CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akVs4CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs4CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akVs4CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs4CaloNegativeCombinedSecondaryVertexBJetTags = akVs4CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akVs4CaloPositiveCombinedSecondaryVertexBJetTags = akVs4CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akVs4CaloSoftPFMuonsTagInfos = akVs4CalobTagger.SoftPFMuonsTagInfos
akVs4CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs4CaloSoftPFMuonBJetTags = akVs4CalobTagger.SoftPFMuonBJetTags
akVs4CaloSoftPFMuonByIP3dBJetTags = akVs4CalobTagger.SoftPFMuonByIP3dBJetTags
akVs4CaloSoftPFMuonByPtBJetTags = akVs4CalobTagger.SoftPFMuonByPtBJetTags
akVs4CaloNegativeSoftPFMuonByPtBJetTags = akVs4CalobTagger.NegativeSoftPFMuonByPtBJetTags
akVs4CaloPositiveSoftPFMuonByPtBJetTags = akVs4CalobTagger.PositiveSoftPFMuonByPtBJetTags
akVs4CaloPatJetFlavourIdLegacy = cms.Sequence(akVs4CaloPatJetPartonAssociationLegacy*akVs4CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs4CaloPatJetFlavourAssociation = akVs4CalobTagger.PatJetFlavourAssociation
#akVs4CaloPatJetFlavourId = cms.Sequence(akVs4CaloPatJetPartons*akVs4CaloPatJetFlavourAssociation)

akVs4CaloJetBtaggingIP       = cms.Sequence(akVs4CaloImpactParameterTagInfos *
            (akVs4CaloTrackCountingHighEffBJetTags +
             akVs4CaloTrackCountingHighPurBJetTags +
             akVs4CaloJetProbabilityBJetTags +
             akVs4CaloJetBProbabilityBJetTags +
             akVs4CaloPositiveOnlyJetProbabilityBJetTags +
             akVs4CaloNegativeOnlyJetProbabilityBJetTags +
             akVs4CaloNegativeTrackCountingHighEffBJetTags +
             akVs4CaloNegativeTrackCountingHighPurBJetTags +
             akVs4CaloNegativeOnlyJetBProbabilityBJetTags +
             akVs4CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs4CaloJetBtaggingSV = cms.Sequence(akVs4CaloImpactParameterTagInfos
            *
            akVs4CaloSecondaryVertexTagInfos
            * (akVs4CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs4CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs4CaloCombinedSecondaryVertexBJetTags
                +
                akVs4CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akVs4CaloJetBtaggingNegSV = cms.Sequence(akVs4CaloImpactParameterTagInfos
            *
            akVs4CaloSecondaryVertexNegativeTagInfos
            * (akVs4CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs4CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs4CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akVs4CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs4CaloJetBtaggingMu = cms.Sequence(akVs4CaloSoftPFMuonsTagInfos * (akVs4CaloSoftPFMuonBJetTags
                +
                akVs4CaloSoftPFMuonByIP3dBJetTags
                +
                akVs4CaloSoftPFMuonByPtBJetTags
                +
                akVs4CaloNegativeSoftPFMuonByPtBJetTags
                +
                akVs4CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs4CaloJetBtagging = cms.Sequence(akVs4CaloJetBtaggingIP
            *akVs4CaloJetBtaggingSV
            *akVs4CaloJetBtaggingNegSV
#            *akVs4CaloJetBtaggingMu
            )

akVs4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs4CaloJets"),
        genJetMatch          = cms.InputTag("akVs4Calomatch"),
        genPartonMatch       = cms.InputTag("akVs4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs4Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs4CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs4CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs4CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs4CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs4CaloJetProbabilityBJetTags"),
            #cms.InputTag("akVs4CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs4CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs4CaloJetID"),
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

akVs4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4GenJets',
                                                             rParam = 0.4,
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
                                                             bTagJetName = cms.untracked.string("akVs4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

akVs4CaloJetSequence_mc = cms.Sequence(
                                                  #akVs4Caloclean
                                                  #*
                                                  akVs4Calomatch
                                                  *
                                                  akVs4Caloparton
                                                  *
                                                  akVs4Calocorr
                                                  *
                                                  #akVs4CaloJetID
                                                  #*
                                                  akVs4CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akVs4CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs4CaloJetBtagging
                                                  *
                                                  akVs4CalopatJetsWithBtagging
                                                  *
                                                  akVs4CaloJetAnalyzer
                                                  )

akVs4CaloJetSequence_data = cms.Sequence(akVs4Calocorr
                                                    *
                                                    #akVs4CaloJetID
                                                    #*
                                                    akVs4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs4CaloJetBtagging
                                                    *
                                                    akVs4CalopatJetsWithBtagging
                                                    *
                                                    akVs4CaloJetAnalyzer
                                                    )

akVs4CaloJetSequence_jec = cms.Sequence(akVs4CaloJetSequence_mc)
akVs4CaloJetSequence_mix = cms.Sequence(akVs4CaloJetSequence_mc)

akVs4CaloJetSequence = cms.Sequence(akVs4CaloJetSequence_data)

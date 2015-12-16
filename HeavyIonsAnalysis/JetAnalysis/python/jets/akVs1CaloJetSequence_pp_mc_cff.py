

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs1CaloJets"),
    matched = cms.InputTag("ak1GenJets"),
    maxDeltaR = 0.1
    )

akVs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs1CaloJets")
                                                        )

akVs1Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs1CaloJets"),
    payload = "AK1Calo_offline"
    )

akVs1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs1CaloJets'))

#akVs1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1GenJets'))

akVs1CalobTagger = bTaggers("akVs1Calo",0.1)

#create objects locally since they dont load properly otherwise
#akVs1Calomatch = akVs1CalobTagger.match
akVs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs1CaloJets"), matched = cms.InputTag("genParticles"))
akVs1CaloPatJetFlavourAssociationLegacy = akVs1CalobTagger.PatJetFlavourAssociationLegacy
akVs1CaloPatJetPartons = akVs1CalobTagger.PatJetPartons
akVs1CaloJetTracksAssociatorAtVertex = akVs1CalobTagger.JetTracksAssociatorAtVertex
akVs1CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs1CaloSimpleSecondaryVertexHighEffBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs1CaloSimpleSecondaryVertexHighPurBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs1CaloCombinedSecondaryVertexBJetTags = akVs1CalobTagger.CombinedSecondaryVertexBJetTags
akVs1CaloCombinedSecondaryVertexV2BJetTags = akVs1CalobTagger.CombinedSecondaryVertexV2BJetTags
akVs1CaloJetBProbabilityBJetTags = akVs1CalobTagger.JetBProbabilityBJetTags
akVs1CaloSoftPFMuonByPtBJetTags = akVs1CalobTagger.SoftPFMuonByPtBJetTags
akVs1CaloSoftPFMuonByIP3dBJetTags = akVs1CalobTagger.SoftPFMuonByIP3dBJetTags
akVs1CaloTrackCountingHighEffBJetTags = akVs1CalobTagger.TrackCountingHighEffBJetTags
akVs1CaloTrackCountingHighPurBJetTags = akVs1CalobTagger.TrackCountingHighPurBJetTags
akVs1CaloPatJetPartonAssociationLegacy = akVs1CalobTagger.PatJetPartonAssociationLegacy

akVs1CaloImpactParameterTagInfos = akVs1CalobTagger.ImpactParameterTagInfos
akVs1CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs1CaloJetProbabilityBJetTags = akVs1CalobTagger.JetProbabilityBJetTags
akVs1CaloPositiveOnlyJetProbabilityBJetTags = akVs1CalobTagger.PositiveOnlyJetProbabilityBJetTags
akVs1CaloNegativeOnlyJetProbabilityBJetTags = akVs1CalobTagger.NegativeOnlyJetProbabilityBJetTags
akVs1CaloNegativeTrackCountingHighEffBJetTags = akVs1CalobTagger.NegativeTrackCountingHighEffBJetTags
akVs1CaloNegativeTrackCountingHighPurBJetTags = akVs1CalobTagger.NegativeTrackCountingHighPurBJetTags
akVs1CaloNegativeOnlyJetBProbabilityBJetTags = akVs1CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akVs1CaloPositiveOnlyJetBProbabilityBJetTags = akVs1CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akVs1CaloSecondaryVertexTagInfos = akVs1CalobTagger.SecondaryVertexTagInfos
akVs1CaloSimpleSecondaryVertexHighEffBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs1CaloSimpleSecondaryVertexHighPurBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs1CaloCombinedSecondaryVertexBJetTags = akVs1CalobTagger.CombinedSecondaryVertexBJetTags
akVs1CaloCombinedSecondaryVertexV2BJetTags = akVs1CalobTagger.CombinedSecondaryVertexV2BJetTags

akVs1CaloSecondaryVertexNegativeTagInfos = akVs1CalobTagger.SecondaryVertexNegativeTagInfos
akVs1CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akVs1CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs1CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akVs1CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs1CaloNegativeCombinedSecondaryVertexBJetTags = akVs1CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akVs1CaloPositiveCombinedSecondaryVertexBJetTags = akVs1CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akVs1CaloSoftPFMuonsTagInfos = akVs1CalobTagger.SoftPFMuonsTagInfos
akVs1CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs1CaloSoftPFMuonBJetTags = akVs1CalobTagger.SoftPFMuonBJetTags
akVs1CaloSoftPFMuonByIP3dBJetTags = akVs1CalobTagger.SoftPFMuonByIP3dBJetTags
akVs1CaloSoftPFMuonByPtBJetTags = akVs1CalobTagger.SoftPFMuonByPtBJetTags
akVs1CaloNegativeSoftPFMuonByPtBJetTags = akVs1CalobTagger.NegativeSoftPFMuonByPtBJetTags
akVs1CaloPositiveSoftPFMuonByPtBJetTags = akVs1CalobTagger.PositiveSoftPFMuonByPtBJetTags
akVs1CaloPatJetFlavourIdLegacy = cms.Sequence(akVs1CaloPatJetPartonAssociationLegacy*akVs1CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs1CaloPatJetFlavourAssociation = akVs1CalobTagger.PatJetFlavourAssociation
#akVs1CaloPatJetFlavourId = cms.Sequence(akVs1CaloPatJetPartons*akVs1CaloPatJetFlavourAssociation)

akVs1CaloJetBtaggingIP       = cms.Sequence(akVs1CaloImpactParameterTagInfos *
            (akVs1CaloTrackCountingHighEffBJetTags +
             akVs1CaloTrackCountingHighPurBJetTags +
             akVs1CaloJetProbabilityBJetTags +
             akVs1CaloJetBProbabilityBJetTags +
             akVs1CaloPositiveOnlyJetProbabilityBJetTags +
             akVs1CaloNegativeOnlyJetProbabilityBJetTags +
             akVs1CaloNegativeTrackCountingHighEffBJetTags +
             akVs1CaloNegativeTrackCountingHighPurBJetTags +
             akVs1CaloNegativeOnlyJetBProbabilityBJetTags +
             akVs1CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs1CaloJetBtaggingSV = cms.Sequence(akVs1CaloImpactParameterTagInfos
            *
            akVs1CaloSecondaryVertexTagInfos
            * (akVs1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs1CaloCombinedSecondaryVertexBJetTags
                +
                akVs1CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akVs1CaloJetBtaggingNegSV = cms.Sequence(akVs1CaloImpactParameterTagInfos
            *
            akVs1CaloSecondaryVertexNegativeTagInfos
            * (akVs1CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs1CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs1CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akVs1CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs1CaloJetBtaggingMu = cms.Sequence(akVs1CaloSoftPFMuonsTagInfos * (akVs1CaloSoftPFMuonBJetTags
                +
                akVs1CaloSoftPFMuonByIP3dBJetTags
                +
                akVs1CaloSoftPFMuonByPtBJetTags
                +
                akVs1CaloNegativeSoftPFMuonByPtBJetTags
                +
                akVs1CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs1CaloJetBtagging = cms.Sequence(akVs1CaloJetBtaggingIP
            *akVs1CaloJetBtaggingSV
            *akVs1CaloJetBtaggingNegSV
#            *akVs1CaloJetBtaggingMu
            )

akVs1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs1CaloJets"),
        genJetMatch          = cms.InputTag("akVs1Calomatch"),
        genPartonMatch       = cms.InputTag("akVs1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs1CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs1CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs1CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs1CaloJetProbabilityBJetTags"),
            #cms.InputTag("akVs1CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs1CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs1CaloJetID"),
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

akVs1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1GenJets',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akVs1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

akVs1CaloJetSequence_mc = cms.Sequence(
                                                  #akVs1Caloclean
                                                  #*
                                                  akVs1Calomatch
                                                  *
                                                  akVs1Caloparton
                                                  *
                                                  akVs1Calocorr
                                                  *
                                                  #akVs1CaloJetID
                                                  #*
                                                  akVs1CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akVs1CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs1CaloJetBtagging
                                                  *
                                                  akVs1CalopatJetsWithBtagging
                                                  *
                                                  akVs1CaloJetAnalyzer
                                                  )

akVs1CaloJetSequence_data = cms.Sequence(akVs1Calocorr
                                                    *
                                                    #akVs1CaloJetID
                                                    #*
                                                    akVs1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs1CaloJetBtagging
                                                    *
                                                    akVs1CalopatJetsWithBtagging
                                                    *
                                                    akVs1CaloJetAnalyzer
                                                    )

akVs1CaloJetSequence_jec = cms.Sequence(akVs1CaloJetSequence_mc)
akVs1CaloJetSequence_mix = cms.Sequence(akVs1CaloJetSequence_mc)

akVs1CaloJetSequence = cms.Sequence(akVs1CaloJetSequence_mc)

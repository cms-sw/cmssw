

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu6CaloJets"),
    matched = cms.InputTag("ak6GenJets"),
    maxDeltaR = 0.6
    )

akPu6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6CaloJets")
                                                        )

akPu6Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu6CaloJets"),
    payload = "AKPu6Calo_offline"
    )

akPu6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu6CaloJets'))

#akPu6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6GenJets'))

akPu6CalobTagger = bTaggers("akPu6Calo",0.6)

#create objects locally since they dont load properly otherwise
#akPu6Calomatch = akPu6CalobTagger.match
akPu6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6CaloJets"), matched = cms.InputTag("genParticles"))
akPu6CaloPatJetFlavourAssociationLegacy = akPu6CalobTagger.PatJetFlavourAssociationLegacy
akPu6CaloPatJetPartons = akPu6CalobTagger.PatJetPartons
akPu6CaloJetTracksAssociatorAtVertex = akPu6CalobTagger.JetTracksAssociatorAtVertex
akPu6CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akPu6CaloSimpleSecondaryVertexHighEffBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6CaloSimpleSecondaryVertexHighPurBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6CaloCombinedSecondaryVertexBJetTags = akPu6CalobTagger.CombinedSecondaryVertexBJetTags
akPu6CaloCombinedSecondaryVertexV2BJetTags = akPu6CalobTagger.CombinedSecondaryVertexV2BJetTags
akPu6CaloJetBProbabilityBJetTags = akPu6CalobTagger.JetBProbabilityBJetTags
akPu6CaloSoftPFMuonByPtBJetTags = akPu6CalobTagger.SoftPFMuonByPtBJetTags
akPu6CaloSoftPFMuonByIP3dBJetTags = akPu6CalobTagger.SoftPFMuonByIP3dBJetTags
akPu6CaloTrackCountingHighEffBJetTags = akPu6CalobTagger.TrackCountingHighEffBJetTags
akPu6CaloTrackCountingHighPurBJetTags = akPu6CalobTagger.TrackCountingHighPurBJetTags
akPu6CaloPatJetPartonAssociationLegacy = akPu6CalobTagger.PatJetPartonAssociationLegacy

akPu6CaloImpactParameterTagInfos = akPu6CalobTagger.ImpactParameterTagInfos
akPu6CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu6CaloJetProbabilityBJetTags = akPu6CalobTagger.JetProbabilityBJetTags
akPu6CaloPositiveOnlyJetProbabilityBJetTags = akPu6CalobTagger.PositiveOnlyJetProbabilityBJetTags
akPu6CaloNegativeOnlyJetProbabilityBJetTags = akPu6CalobTagger.NegativeOnlyJetProbabilityBJetTags
akPu6CaloNegativeTrackCountingHighEffBJetTags = akPu6CalobTagger.NegativeTrackCountingHighEffBJetTags
akPu6CaloNegativeTrackCountingHighPurBJetTags = akPu6CalobTagger.NegativeTrackCountingHighPurBJetTags
akPu6CaloNegativeOnlyJetBProbabilityBJetTags = akPu6CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akPu6CaloPositiveOnlyJetBProbabilityBJetTags = akPu6CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akPu6CaloSecondaryVertexTagInfos = akPu6CalobTagger.SecondaryVertexTagInfos
akPu6CaloSimpleSecondaryVertexHighEffBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6CaloSimpleSecondaryVertexHighPurBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6CaloCombinedSecondaryVertexBJetTags = akPu6CalobTagger.CombinedSecondaryVertexBJetTags
akPu6CaloCombinedSecondaryVertexV2BJetTags = akPu6CalobTagger.CombinedSecondaryVertexV2BJetTags

akPu6CaloSecondaryVertexNegativeTagInfos = akPu6CalobTagger.SecondaryVertexNegativeTagInfos
akPu6CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akPu6CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu6CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akPu6CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu6CaloNegativeCombinedSecondaryVertexBJetTags = akPu6CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akPu6CaloPositiveCombinedSecondaryVertexBJetTags = akPu6CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akPu6CaloSoftPFMuonsTagInfos = akPu6CalobTagger.SoftPFMuonsTagInfos
akPu6CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu6CaloSoftPFMuonBJetTags = akPu6CalobTagger.SoftPFMuonBJetTags
akPu6CaloSoftPFMuonByIP3dBJetTags = akPu6CalobTagger.SoftPFMuonByIP3dBJetTags
akPu6CaloSoftPFMuonByPtBJetTags = akPu6CalobTagger.SoftPFMuonByPtBJetTags
akPu6CaloNegativeSoftPFMuonByPtBJetTags = akPu6CalobTagger.NegativeSoftPFMuonByPtBJetTags
akPu6CaloPositiveSoftPFMuonByPtBJetTags = akPu6CalobTagger.PositiveSoftPFMuonByPtBJetTags
akPu6CaloPatJetFlavourIdLegacy = cms.Sequence(akPu6CaloPatJetPartonAssociationLegacy*akPu6CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu6CaloPatJetFlavourAssociation = akPu6CalobTagger.PatJetFlavourAssociation
#akPu6CaloPatJetFlavourId = cms.Sequence(akPu6CaloPatJetPartons*akPu6CaloPatJetFlavourAssociation)

akPu6CaloJetBtaggingIP       = cms.Sequence(akPu6CaloImpactParameterTagInfos *
            (akPu6CaloTrackCountingHighEffBJetTags +
             akPu6CaloTrackCountingHighPurBJetTags +
             akPu6CaloJetProbabilityBJetTags +
             akPu6CaloJetBProbabilityBJetTags +
             akPu6CaloPositiveOnlyJetProbabilityBJetTags +
             akPu6CaloNegativeOnlyJetProbabilityBJetTags +
             akPu6CaloNegativeTrackCountingHighEffBJetTags +
             akPu6CaloNegativeTrackCountingHighPurBJetTags +
             akPu6CaloNegativeOnlyJetBProbabilityBJetTags +
             akPu6CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akPu6CaloJetBtaggingSV = cms.Sequence(akPu6CaloImpactParameterTagInfos
            *
            akPu6CaloSecondaryVertexTagInfos
            * (akPu6CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu6CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu6CaloCombinedSecondaryVertexBJetTags
                +
                akPu6CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akPu6CaloJetBtaggingNegSV = cms.Sequence(akPu6CaloImpactParameterTagInfos
            *
            akPu6CaloSecondaryVertexNegativeTagInfos
            * (akPu6CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu6CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu6CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akPu6CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu6CaloJetBtaggingMu = cms.Sequence(akPu6CaloSoftPFMuonsTagInfos * (akPu6CaloSoftPFMuonBJetTags
                +
                akPu6CaloSoftPFMuonByIP3dBJetTags
                +
                akPu6CaloSoftPFMuonByPtBJetTags
                +
                akPu6CaloNegativeSoftPFMuonByPtBJetTags
                +
                akPu6CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akPu6CaloJetBtagging = cms.Sequence(akPu6CaloJetBtaggingIP
            *akPu6CaloJetBtaggingSV
            *akPu6CaloJetBtaggingNegSV
#            *akPu6CaloJetBtaggingMu
            )

akPu6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu6CaloJets"),
        genJetMatch          = cms.InputTag("akPu6Calomatch"),
        genPartonMatch       = cms.InputTag("akPu6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu6Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu6CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu6CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu6CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu6CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu6CaloJetProbabilityBJetTags"),
            #cms.InputTag("akPu6CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akPu6CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akPu6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu6CaloJetID"),
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

akPu6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu6CalopatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akPu6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

akPu6CaloJetSequence_mc = cms.Sequence(
                                                  #akPu6Caloclean
                                                  #*
                                                  akPu6Calomatch
                                                  *
                                                  akPu6Caloparton
                                                  *
                                                  akPu6Calocorr
                                                  *
                                                  #akPu6CaloJetID
                                                  #*
                                                  akPu6CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akPu6CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akPu6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu6CaloJetBtagging
                                                  *
                                                  akPu6CalopatJetsWithBtagging
                                                  *
                                                  akPu6CaloJetAnalyzer
                                                  )

akPu6CaloJetSequence_data = cms.Sequence(akPu6Calocorr
                                                    *
                                                    #akPu6CaloJetID
                                                    #*
                                                    akPu6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu6CaloJetBtagging
                                                    *
                                                    akPu6CalopatJetsWithBtagging
                                                    *
                                                    akPu6CaloJetAnalyzer
                                                    )

akPu6CaloJetSequence_jec = cms.Sequence(akPu6CaloJetSequence_mc)
akPu6CaloJetSequence_mix = cms.Sequence(akPu6CaloJetSequence_mc)

akPu6CaloJetSequence = cms.Sequence(akPu6CaloJetSequence_mc)

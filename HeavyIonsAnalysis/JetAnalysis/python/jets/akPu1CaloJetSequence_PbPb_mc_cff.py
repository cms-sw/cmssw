

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets"),
    maxDeltaR = 0.1
    )

akPu1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu1CaloJets")
                                                        )

akPu1Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu1CaloJets"),
    payload = "AKPu1Calo_offline"
    )

akPu1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu1CaloJets'))

#akPu1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJets'))

akPu1CalobTagger = bTaggers("akPu1Calo",0.1)

#create objects locally since they dont load properly otherwise
#akPu1Calomatch = akPu1CalobTagger.match
akPu1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu1CaloJets"), matched = cms.InputTag("genParticles"))
akPu1CaloPatJetFlavourAssociationLegacy = akPu1CalobTagger.PatJetFlavourAssociationLegacy
akPu1CaloPatJetPartons = akPu1CalobTagger.PatJetPartons
akPu1CaloJetTracksAssociatorAtVertex = akPu1CalobTagger.JetTracksAssociatorAtVertex
akPu1CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akPu1CaloSimpleSecondaryVertexHighEffBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu1CaloSimpleSecondaryVertexHighPurBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu1CaloCombinedSecondaryVertexBJetTags = akPu1CalobTagger.CombinedSecondaryVertexBJetTags
akPu1CaloCombinedSecondaryVertexV2BJetTags = akPu1CalobTagger.CombinedSecondaryVertexV2BJetTags
akPu1CaloJetBProbabilityBJetTags = akPu1CalobTagger.JetBProbabilityBJetTags
akPu1CaloSoftPFMuonByPtBJetTags = akPu1CalobTagger.SoftPFMuonByPtBJetTags
akPu1CaloSoftPFMuonByIP3dBJetTags = akPu1CalobTagger.SoftPFMuonByIP3dBJetTags
akPu1CaloTrackCountingHighEffBJetTags = akPu1CalobTagger.TrackCountingHighEffBJetTags
akPu1CaloTrackCountingHighPurBJetTags = akPu1CalobTagger.TrackCountingHighPurBJetTags
akPu1CaloPatJetPartonAssociationLegacy = akPu1CalobTagger.PatJetPartonAssociationLegacy

akPu1CaloImpactParameterTagInfos = akPu1CalobTagger.ImpactParameterTagInfos
akPu1CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu1CaloJetProbabilityBJetTags = akPu1CalobTagger.JetProbabilityBJetTags
akPu1CaloPositiveOnlyJetProbabilityBJetTags = akPu1CalobTagger.PositiveOnlyJetProbabilityBJetTags
akPu1CaloNegativeOnlyJetProbabilityBJetTags = akPu1CalobTagger.NegativeOnlyJetProbabilityBJetTags
akPu1CaloNegativeTrackCountingHighEffBJetTags = akPu1CalobTagger.NegativeTrackCountingHighEffBJetTags
akPu1CaloNegativeTrackCountingHighPurBJetTags = akPu1CalobTagger.NegativeTrackCountingHighPurBJetTags
akPu1CaloNegativeOnlyJetBProbabilityBJetTags = akPu1CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akPu1CaloPositiveOnlyJetBProbabilityBJetTags = akPu1CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akPu1CaloSecondaryVertexTagInfos = akPu1CalobTagger.SecondaryVertexTagInfos
akPu1CaloSimpleSecondaryVertexHighEffBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu1CaloSimpleSecondaryVertexHighPurBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu1CaloCombinedSecondaryVertexBJetTags = akPu1CalobTagger.CombinedSecondaryVertexBJetTags
akPu1CaloCombinedSecondaryVertexV2BJetTags = akPu1CalobTagger.CombinedSecondaryVertexV2BJetTags

akPu1CaloSecondaryVertexNegativeTagInfos = akPu1CalobTagger.SecondaryVertexNegativeTagInfos
akPu1CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akPu1CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu1CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akPu1CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu1CaloNegativeCombinedSecondaryVertexBJetTags = akPu1CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akPu1CaloPositiveCombinedSecondaryVertexBJetTags = akPu1CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akPu1CaloSoftPFMuonsTagInfos = akPu1CalobTagger.SoftPFMuonsTagInfos
akPu1CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu1CaloSoftPFMuonBJetTags = akPu1CalobTagger.SoftPFMuonBJetTags
akPu1CaloSoftPFMuonByIP3dBJetTags = akPu1CalobTagger.SoftPFMuonByIP3dBJetTags
akPu1CaloSoftPFMuonByPtBJetTags = akPu1CalobTagger.SoftPFMuonByPtBJetTags
akPu1CaloNegativeSoftPFMuonByPtBJetTags = akPu1CalobTagger.NegativeSoftPFMuonByPtBJetTags
akPu1CaloPositiveSoftPFMuonByPtBJetTags = akPu1CalobTagger.PositiveSoftPFMuonByPtBJetTags
akPu1CaloPatJetFlavourIdLegacy = cms.Sequence(akPu1CaloPatJetPartonAssociationLegacy*akPu1CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu1CaloPatJetFlavourAssociation = akPu1CalobTagger.PatJetFlavourAssociation
#akPu1CaloPatJetFlavourId = cms.Sequence(akPu1CaloPatJetPartons*akPu1CaloPatJetFlavourAssociation)

akPu1CaloJetBtaggingIP       = cms.Sequence(akPu1CaloImpactParameterTagInfos *
            (akPu1CaloTrackCountingHighEffBJetTags +
             akPu1CaloTrackCountingHighPurBJetTags +
             akPu1CaloJetProbabilityBJetTags +
             akPu1CaloJetBProbabilityBJetTags +
             akPu1CaloPositiveOnlyJetProbabilityBJetTags +
             akPu1CaloNegativeOnlyJetProbabilityBJetTags +
             akPu1CaloNegativeTrackCountingHighEffBJetTags +
             akPu1CaloNegativeTrackCountingHighPurBJetTags +
             akPu1CaloNegativeOnlyJetBProbabilityBJetTags +
             akPu1CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akPu1CaloJetBtaggingSV = cms.Sequence(akPu1CaloImpactParameterTagInfos
            *
            akPu1CaloSecondaryVertexTagInfos
            * (akPu1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu1CaloCombinedSecondaryVertexBJetTags
                +
                akPu1CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akPu1CaloJetBtaggingNegSV = cms.Sequence(akPu1CaloImpactParameterTagInfos
            *
            akPu1CaloSecondaryVertexNegativeTagInfos
            * (akPu1CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu1CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu1CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akPu1CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu1CaloJetBtaggingMu = cms.Sequence(akPu1CaloSoftPFMuonsTagInfos * (akPu1CaloSoftPFMuonBJetTags
                +
                akPu1CaloSoftPFMuonByIP3dBJetTags
                +
                akPu1CaloSoftPFMuonByPtBJetTags
                +
                akPu1CaloNegativeSoftPFMuonByPtBJetTags
                +
                akPu1CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akPu1CaloJetBtagging = cms.Sequence(akPu1CaloJetBtaggingIP
            *akPu1CaloJetBtaggingSV
            *akPu1CaloJetBtaggingNegSV
#            *akPu1CaloJetBtaggingMu
            )

akPu1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu1CaloJets"),
        genJetMatch          = cms.InputTag("akPu1Calomatch"),
        genPartonMatch       = cms.InputTag("akPu1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu1CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu1CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu1CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu1CaloJetProbabilityBJetTags"),
            #cms.InputTag("akPu1CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akPu1CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akPu1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu1CaloJetID"),
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

akPu1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akPu1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akPu1CaloJetSequence_mc = cms.Sequence(
                                                  #akPu1Caloclean
                                                  #*
                                                  akPu1Calomatch
                                                  *
                                                  akPu1Caloparton
                                                  *
                                                  akPu1Calocorr
                                                  *
                                                  #akPu1CaloJetID
                                                  #*
                                                  akPu1CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akPu1CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akPu1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu1CaloJetBtagging
                                                  *
                                                  akPu1CalopatJetsWithBtagging
                                                  *
                                                  akPu1CaloJetAnalyzer
                                                  )

akPu1CaloJetSequence_data = cms.Sequence(akPu1Calocorr
                                                    *
                                                    #akPu1CaloJetID
                                                    #*
                                                    akPu1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu1CaloJetBtagging
                                                    *
                                                    akPu1CalopatJetsWithBtagging
                                                    *
                                                    akPu1CaloJetAnalyzer
                                                    )

akPu1CaloJetSequence_jec = cms.Sequence(akPu1CaloJetSequence_mc)
akPu1CaloJetSequence_mix = cms.Sequence(akPu1CaloJetSequence_mc)

akPu1CaloJetSequence = cms.Sequence(akPu1CaloJetSequence_mc)

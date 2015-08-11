

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs7CaloJets"),
    matched = cms.InputTag("ak7HiGenJets"),
    maxDeltaR = 0.7
    )

akVs7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs7CaloJets")
                                                        )

akVs7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs7CaloJets"),
    payload = "AKVs7Calo_HI"
    )

akVs7CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs7CaloJets'))

#akVs7Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

akVs7CalobTagger = bTaggers("akVs7Calo",0.7)

#create objects locally since they dont load properly otherwise
#akVs7Calomatch = akVs7CalobTagger.match
akVs7Caloparton = akVs7CalobTagger.parton
akVs7CaloPatJetFlavourAssociationLegacy = akVs7CalobTagger.PatJetFlavourAssociationLegacy
akVs7CaloPatJetPartons = akVs7CalobTagger.PatJetPartons
akVs7CaloJetTracksAssociatorAtVertex = akVs7CalobTagger.JetTracksAssociatorAtVertex
akVs7CaloSimpleSecondaryVertexHighEffBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7CaloSimpleSecondaryVertexHighPurBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7CaloCombinedSecondaryVertexBJetTags = akVs7CalobTagger.CombinedSecondaryVertexBJetTags
akVs7CaloCombinedSecondaryVertexV2BJetTags = akVs7CalobTagger.CombinedSecondaryVertexV2BJetTags
akVs7CaloJetBProbabilityBJetTags = akVs7CalobTagger.JetBProbabilityBJetTags
akVs7CaloSoftPFMuonByPtBJetTags = akVs7CalobTagger.SoftPFMuonByPtBJetTags
akVs7CaloSoftPFMuonByIP3dBJetTags = akVs7CalobTagger.SoftPFMuonByIP3dBJetTags
akVs7CaloTrackCountingHighEffBJetTags = akVs7CalobTagger.TrackCountingHighEffBJetTags
akVs7CaloTrackCountingHighPurBJetTags = akVs7CalobTagger.TrackCountingHighPurBJetTags
akVs7CaloPatJetPartonAssociationLegacy = akVs7CalobTagger.PatJetPartonAssociationLegacy

akVs7CaloImpactParameterTagInfos = akVs7CalobTagger.ImpactParameterTagInfos
akVs7CaloJetProbabilityBJetTags = akVs7CalobTagger.JetProbabilityBJetTags
akVs7CaloPositiveOnlyJetProbabilityBJetTags = akVs7CalobTagger.PositiveOnlyJetProbabilityBJetTags
akVs7CaloNegativeOnlyJetProbabilityBJetTags = akVs7CalobTagger.NegativeOnlyJetProbabilityBJetTags
akVs7CaloNegativeTrackCountingHighEffBJetTags = akVs7CalobTagger.NegativeTrackCountingHighEffBJetTags
akVs7CaloNegativeTrackCountingHighPurBJetTags = akVs7CalobTagger.NegativeTrackCountingHighPurBJetTags
akVs7CaloNegativeOnlyJetBProbabilityBJetTags = akVs7CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akVs7CaloPositiveOnlyJetBProbabilityBJetTags = akVs7CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akVs7CaloSecondaryVertexTagInfos = akVs7CalobTagger.SecondaryVertexTagInfos
akVs7CaloSimpleSecondaryVertexHighEffBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7CaloSimpleSecondaryVertexHighPurBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7CaloCombinedSecondaryVertexBJetTags = akVs7CalobTagger.CombinedSecondaryVertexBJetTags
akVs7CaloCombinedSecondaryVertexV2BJetTags = akVs7CalobTagger.CombinedSecondaryVertexV2BJetTags

akVs7CaloSecondaryVertexNegativeTagInfos = akVs7CalobTagger.SecondaryVertexNegativeTagInfos
akVs7CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akVs7CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs7CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akVs7CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs7CaloNegativeCombinedSecondaryVertexBJetTags = akVs7CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akVs7CaloPositiveCombinedSecondaryVertexBJetTags = akVs7CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akVs7CaloSoftPFMuonsTagInfos = akVs7CalobTagger.SoftPFMuonsTagInfos
akVs7CaloSoftPFMuonBJetTags = akVs7CalobTagger.SoftPFMuonBJetTags
akVs7CaloSoftPFMuonByIP3dBJetTags = akVs7CalobTagger.SoftPFMuonByIP3dBJetTags
akVs7CaloSoftPFMuonByPtBJetTags = akVs7CalobTagger.SoftPFMuonByPtBJetTags
akVs7CaloNegativeSoftPFMuonByPtBJetTags = akVs7CalobTagger.NegativeSoftPFMuonByPtBJetTags
akVs7CaloPositiveSoftPFMuonByPtBJetTags = akVs7CalobTagger.PositiveSoftPFMuonByPtBJetTags
akVs7CaloPatJetFlavourIdLegacy = cms.Sequence(akVs7CaloPatJetPartonAssociationLegacy*akVs7CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs7CaloPatJetFlavourAssociation = akVs7CalobTagger.PatJetFlavourAssociation
#akVs7CaloPatJetFlavourId = cms.Sequence(akVs7CaloPatJetPartons*akVs7CaloPatJetFlavourAssociation)

akVs7CaloJetBtaggingIP       = cms.Sequence(akVs7CaloImpactParameterTagInfos *
            (akVs7CaloTrackCountingHighEffBJetTags +
             akVs7CaloTrackCountingHighPurBJetTags +
             akVs7CaloJetProbabilityBJetTags +
             akVs7CaloJetBProbabilityBJetTags +
             akVs7CaloPositiveOnlyJetProbabilityBJetTags +
             akVs7CaloNegativeOnlyJetProbabilityBJetTags +
             akVs7CaloNegativeTrackCountingHighEffBJetTags +
             akVs7CaloNegativeTrackCountingHighPurBJetTags +
             akVs7CaloNegativeOnlyJetBProbabilityBJetTags +
             akVs7CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs7CaloJetBtaggingSV = cms.Sequence(akVs7CaloImpactParameterTagInfos
            *
            akVs7CaloSecondaryVertexTagInfos
            * (akVs7CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs7CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs7CaloCombinedSecondaryVertexBJetTags
                +
                akVs7CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akVs7CaloJetBtaggingNegSV = cms.Sequence(akVs7CaloImpactParameterTagInfos
            *
            akVs7CaloSecondaryVertexNegativeTagInfos
            * (akVs7CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs7CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs7CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akVs7CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs7CaloJetBtaggingMu = cms.Sequence(akVs7CaloSoftPFMuonsTagInfos * (akVs7CaloSoftPFMuonBJetTags
                +
                akVs7CaloSoftPFMuonByIP3dBJetTags
                +
                akVs7CaloSoftPFMuonByPtBJetTags
                +
                akVs7CaloNegativeSoftPFMuonByPtBJetTags
                +
                akVs7CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs7CaloJetBtagging = cms.Sequence(akVs7CaloJetBtaggingIP
            *akVs7CaloJetBtaggingSV
            *akVs7CaloJetBtaggingNegSV
            *akVs7CaloJetBtaggingMu
            )

akVs7CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs7CaloJets"),
        genJetMatch          = cms.InputTag("akVs7Calomatch"),
        genPartonMatch       = cms.InputTag("akVs7Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs7Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs7CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs7CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs7CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs7CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs7CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs7CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs7CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs7CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs7CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs7CaloSoftPFMuonByPtBJetTags"),
            cms.InputTag("akVs7CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs7CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs7CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs7CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akVs7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs7CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs7Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs7CaloJetSequence_mc = cms.Sequence(
                                                  #akVs7Caloclean
                                                  #*
                                                  akVs7Calomatch
                                                  *
                                                  akVs7Caloparton
                                                  *
                                                  akVs7Calocorr
                                                  *
                                                  akVs7CaloJetID
                                                  *
                                                  akVs7CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akVs7CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs7CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs7CaloJetBtagging
                                                  *
                                                  akVs7CalopatJetsWithBtagging
                                                  *
                                                  akVs7CaloJetAnalyzer
                                                  )

akVs7CaloJetSequence_data = cms.Sequence(akVs7Calocorr
                                                    *
                                                    akVs7CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs7CaloJetBtagging
                                                    *
                                                    akVs7CalopatJetsWithBtagging
                                                    *
                                                    akVs7CaloJetAnalyzer
                                                    )

akVs7CaloJetSequence_jec = cms.Sequence(akVs7CaloJetSequence_mc)
akVs7CaloJetSequence_mix = cms.Sequence(akVs7CaloJetSequence_mc)

akVs7CaloJetSequence = cms.Sequence(akVs7CaloJetSequence_data)
